//! Image recognition: local image -> bounded downscale -> multimodal model -> plain text.
//!
//! Modeled on `vision-tool` from `my-deepseek-harness`: shrink the long edge to `max_edge`
//! before upload to keep pixel-billed cost down; detect the format from magic bytes rather
//! than the extension; include each image's path and sha in the output header so multi-turn
//! conversations can map results back to the original file.
//!
//! The tool layer returns text only, so callers (the understanding agent) never have to deal
//! with multimodal messages themselves.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use image::{
    codecs::jpeg::JpegEncoder, imageops::FilterType, DynamicImage, GenericImageView, ImageFormat,
};
use relay_openai::{CompletionRequest, InputItem, LanguageModel};
use sha2::{Digest, Sha256};

/// Long edge used when the caller does not specify one.
pub const DEFAULT_MAX_EDGE: u32 = 1024;
/// Allowed `max_edge` values: from token-saving to small-text-preserving.
pub const ALLOWED_MAX_EDGES: [u32; 5] = [768, 960, 1024, 1280, 1568];
/// Maximum number of images per call.
pub const MAX_IMAGES: usize = 3;

/// Upper bound when reading the original from disk; originals are downscaled anyway, so this
/// only guards against runaway sizes.
const MAX_SOURCE_BYTES: u64 = 64 * 1024 * 1024;
/// Upper bound on actual upload bytes per image (after downscaling).
const MAX_UPLOAD_BYTES: usize = 10 * 1024 * 1024;
/// Pixel limit enforced before decoding, so crafted images cannot exhaust memory.
const MAX_PIXELS: u64 = 64_000_000;
/// Upper bound on the recognized text returned to the caller.
const MAX_OUTPUT_CHARS: usize = 20_000;
/// Lossy re-encode quality.
const JPEG_QUALITY: u8 = 80;

/// Fixed system prompt: extract only, never improvise, predictable output shape.
const SYSTEM_PROMPT: &str =
    "You are an image recognition assistant. Extract only the information this task needs.\n\n\
1. Recognize only relevant content and ignore unrelated details.\n\
2. Output the result directly; no explanation, reasoning, or redundant description.\n\
3. Extract text and numbers exactly; do not guess or complete them.\n\
4. Mark illegible content as \"illegible\".\n\
5. Keep it within 100 characters by default; full-extraction tasks have no length limit.\n\n\
Return only the recognition result.";

/// An image prepared for upload.
#[derive(Debug, Clone)]
pub struct PreparedImage {
    /// Original file path, echoed in the output header for traceability.
    pub path: String,
    pub mime: &'static str,
    pub bytes: Vec<u8>,
    /// First 8 hex digits of the sha256 over the uploaded bytes.
    pub sha: String,
    /// Dimensions before and after downscaling; `None` when not resized.
    pub downscaled: Option<((u32, u32), (u32, u32))>,
}

/// Detect the image format from magic bytes; never trust the extension.
pub fn sniff_mime(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
        Some("image/png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF8") {
        Some("image/gif")
    } else if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else if bytes.starts_with(b"BM") {
        Some("image/bmp")
    } else {
        None
    }
}

fn image_format(mime: &str) -> Option<ImageFormat> {
    Some(match mime {
        "image/png" => ImageFormat::Png,
        "image/jpeg" => ImageFormat::Jpeg,
        "image/gif" => ImageFormat::Gif,
        "image/webp" => ImageFormat::WebP,
        "image/bmp" => ImageFormat::Bmp,
        _ => return None,
    })
}

fn short_sha(bytes: &[u8]) -> String {
    let hex = format!("{:x}", Sha256::digest(bytes));
    hex[..8].to_string()
}

/// Read and (if needed) downscale to `max_edge`. Files whose long edge already fits are
/// uploaded as-is, so lossless screenshots are not re-encoded for nothing.
pub fn prepare(path: &Path, max_edge: u32) -> Result<PreparedImage, String> {
    let size = std::fs::metadata(path)
        .map_err(|err| format!("read failed: {err}"))?
        .len();
    if size == 0 {
        return Err("image is empty".into());
    }
    if size > MAX_SOURCE_BYTES {
        return Err(format!(
            "source image too large ({} MB, limit 64 MB)",
            size / 1024 / 1024
        ));
    }
    let bytes = std::fs::read(path).map_err(|err| format!("read failed: {err}"))?;
    let mime = sniff_mime(&bytes).ok_or_else(|| {
        "unsupported image format (supported: png / jpg / gif / webp / bmp)".to_string()
    })?;
    let format = image_format(mime).ok_or_else(|| "unsupported image format".to_string())?;

    // Read only the header for dimensions, so oversized images are never decoded first.
    let (width, height) = image::ImageReader::with_format(Cursor::new(&bytes), format)
        .into_dimensions()
        .map_err(|err| format!("cannot parse image: {err}"))?;
    if u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(format!(
            "{width}x{height} pixels exceeds the 64-megapixel limit"
        ));
    }

    if width.max(height) <= max_edge {
        if bytes.len() > MAX_UPLOAD_BYTES {
            return Err(format!(
                "image is {} MB, over the 10 MB per-image limit",
                bytes.len() / 1024 / 1024
            ));
        }
        return Ok(PreparedImage {
            path: path.display().to_string(),
            mime,
            sha: short_sha(&bytes),
            bytes,
            downscaled: None,
        });
    }

    let image = image::load_from_memory_with_format(&bytes, format)
        .map_err(|err| format!("cannot decode image: {err}"))?;
    let resized = image.resize(max_edge, max_edge, FilterType::Lanczos3);
    let (bytes, mime) = encode(&resized)?;
    if bytes.len() > MAX_UPLOAD_BYTES {
        return Err("still over the 10 MB per-image limit after downscaling".into());
    }
    Ok(PreparedImage {
        path: path.display().to_string(),
        mime,
        sha: short_sha(&bytes),
        downscaled: Some(((width, height), resized.dimensions())),
        bytes,
    })
}

/// Use PNG when there is an alpha channel (keeps text crisp), otherwise JPEG for photo-like
/// content.
fn encode(image: &DynamicImage) -> Result<(Vec<u8>, &'static str), String> {
    if image.color().has_alpha() {
        let mut out = Vec::new();
        image
            .write_to(&mut Cursor::new(&mut out), ImageFormat::Png)
            .map_err(|err| format!("re-encode failed: {err}"))?;
        Ok((out, "image/png"))
    } else {
        let rgb = image.to_rgb8();
        let mut out = Vec::new();
        JpegEncoder::new_with_quality(&mut out, JPEG_QUALITY)
            .encode_image(&rgb)
            .map_err(|err| format!("re-encode failed: {err}"))?;
        Ok((out, "image/jpeg"))
    }
}

/// Send the images together with the question to the model and return the image header plus
/// the recognition result.
pub async fn describe(
    model: &dyn LanguageModel,
    images: &[PreparedImage],
    prompt: &str,
    max_edge: u32,
    model_name: Option<&str>,
) -> Result<String, String> {
    let mut items = Vec::with_capacity(images.len() + 2);
    if images.len() > 1 {
        let listing = images
            .iter()
            .enumerate()
            .map(|(index, image)| format!("{}. {}", index + 1, image.path))
            .collect::<Vec<_>>()
            .join("\n");
        items.push(InputItem::Text(format!(
            "The following question refers to {} images, attached in this order:\n{listing}",
            images.len()
        )));
    }
    items.push(InputItem::Text(prompt.to_string()));
    for image in images {
        items.push(InputItem::ImageBytes {
            mime: image.mime.to_string(),
            bytes: image.bytes.clone(),
        });
    }

    let request = CompletionRequest {
        instructions: SYSTEM_PROMPT.to_string(),
        input: items,
        model: model_name.map(str::to_string),
        temperature: Some(0.0),
        max_output_tokens: 2000,
        previous_response_id: None,
        tools: Vec::new(),
    };
    let output = model
        .complete(request)
        .await
        .map_err(|err| format!("{err:#}"))?;
    let text = output.text.trim();
    if text.is_empty() {
        return Err("the model returned no recognized content (the current model may not support image input)".into());
    }

    let header = images
        .iter()
        .map(|image| {
            format!(
                "image\t{}\tsha={}\tmax_edge={max_edge}",
                image.path, image.sha
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let (answer, truncated) = bounded(text, MAX_OUTPUT_CHARS);
    if truncated {
        Ok(format!(
            "{header}\n{answer}\n\n[recognition result exceeded {MAX_OUTPUT_CHARS} characters and was truncated]"
        ))
    } else {
        Ok(format!("{header}\n{answer}"))
    }
}

/// Validate a set of paths (argument-level checks only: presence, format, count and size).
pub fn validate_paths(paths: &[PathBuf]) -> Result<(), String> {
    if paths.is_empty() {
        return Err("path must not be empty".into());
    }
    if paths.len() > MAX_IMAGES {
        return Err(format!("at most {MAX_IMAGES} images per call"));
    }
    if paths.iter().any(|path| path.as_os_str().is_empty()) {
        return Err("path must not contain empty entries".into());
    }
    Ok(())
}

pub fn validate_max_edge(max_edge: u32) -> Result<(), String> {
    if ALLOWED_MAX_EDGES.contains(&max_edge) {
        Ok(())
    } else {
        Err(format!(
            "max_edge must be one of {}",
            ALLOWED_MAX_EDGES
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(" / ")
        ))
    }
}

fn bounded(text: &str, max: usize) -> (String, bool) {
    if text.chars().count() <= max {
        (text.to_string(), false)
    } else {
        (text.chars().take(max).collect(), true)
    }
}

#[cfg(test)]
mod tests {
    use super::{prepare, sniff_mime};
    use image::{Rgba, RgbaImage};

    #[test]
    fn sniffs_formats_from_magic_bytes() {
        assert_eq!(sniff_mime(b"\x89PNG\r\n\x1a\nrest"), Some("image/png"));
        assert_eq!(sniff_mime(b"\xFF\xD8\xFFrest"), Some("image/jpeg"));
        assert_eq!(sniff_mime(b"GIF89a"), Some("image/gif"));
        assert_eq!(sniff_mime(b"RIFF\0\0\0\0WEBPVP8 "), Some("image/webp"));
        assert_eq!(sniff_mime(b"BMrest"), Some("image/bmp"));
        assert_eq!(sniff_mime(b"plain text"), None);
    }

    #[test]
    fn downscales_long_edge_and_keeps_aspect_ratio() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.png");
        RgbaImage::from_pixel(2000, 1000, Rgba([255, 255, 255, 255]))
            .save(&path)
            .unwrap();

        let prepared = prepare(&path, 1024).unwrap();
        assert_eq!(
            prepared.downscaled,
            Some(((2000, 1000), (1024, 512))),
            "long edge shrunk to 1024, aspect ratio preserved"
        );
        assert_eq!(
            prepared.mime, "image/png",
            "images with alpha are re-encoded as PNG"
        );
        assert_eq!(prepared.sha.len(), 8);
    }

    #[test]
    fn small_images_are_forwarded_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.png");
        RgbaImage::from_pixel(320, 200, Rgba([10, 20, 30, 255]))
            .save(&path)
            .unwrap();
        let before = std::fs::read(&path).unwrap();

        let prepared = prepare(&path, 1024).unwrap();
        assert!(prepared.downscaled.is_none());
        assert_eq!(
            prepared.bytes, before,
            "images within the limits are uploaded untouched"
        );
    }
}
