//! Smoke tests for cross-platform context: permissions are not required, but capture
//! must not panic and the snapshot must stay self-consistent.

use relay_context::{CaptureOptions, ContextProvider, NativeContextProvider};
use relay_core::ContextSource;

#[test]
fn capture_returns_native_snapshot_without_panicking() {
    let provider = NativeContextProvider::new();
    let options = CaptureOptions {
        include_clipboard: true,
        clipboard_max_chars: 200,
        ..Default::default()
    };
    let snapshot = provider.capture(&options).expect("capture should not fail");
    assert_eq!(snapshot.source, ContextSource::Native);
    assert!(snapshot.captured_at.is_some());
    if let Some(clip) = snapshot.clipboard_excerpt.as_deref() {
        assert!(clip.chars().count() <= 220);
    }
}

#[test]
fn screenshot_option_writes_or_reports_permission() {
    let dir = tempfile::tempdir().unwrap();
    let provider = NativeContextProvider::new();
    let options = CaptureOptions {
        include_screenshot: true,
        screenshot_dir: Some(dir.path().to_path_buf()),
        ..Default::default()
    };
    let snapshot = provider.capture(&options).expect("capture should not fail");
    // With screen recording permission there should be screenshots; without it they are
    // simply skipped instead of failing.
    for path in &snapshot.screenshots {
        assert!(path.starts_with(dir.path()));
        assert!(std::fs::metadata(path).unwrap().len() > 0);
    }
}
