/// Lightweight tokenizer: splits on non-alphanumeric characters and additionally breaks CJK runs
/// into bigrams, so window titles can be matched against keywords.
///
/// Key detail: in mixed CJK / digit / letter text the CJK runs must be **split out first**,
/// otherwise the whole chunk is treated as one word and CJK terms never match.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();

    for chunk in text.split(|ch: char| !(ch.is_alphanumeric() || is_cjk(ch))) {
        if chunk.is_empty() {
            continue;
        }
        // First split the chunk into "CJK runs" and "non-CJK runs".
        let mut cjk = String::new();
        let mut other = String::new();
        let flush_cjk = |cjk: &mut String, tokens: &mut Vec<String>| {
            if cjk.is_empty() {
                return;
            }
            let word = cjk.to_lowercase();
            if word.chars().count() > 2 {
                let chars: Vec<char> = word.chars().collect();
                for window in chars.windows(2) {
                    tokens.push(window.iter().collect());
                }
            }
            tokens.push(word);
            cjk.clear();
        };
        let flush_other = |other: &mut String, tokens: &mut Vec<String>| {
            if !other.is_empty() {
                tokens.push(other.to_lowercase());
                other.clear();
            }
        };
        for ch in chunk.chars() {
            if is_cjk(ch) {
                flush_other(&mut other, &mut tokens);
                cjk.push(ch);
            } else {
                flush_cjk(&mut cjk, &mut tokens);
                other.push(ch);
            }
        }
        flush_cjk(&mut cjk, &mut tokens);
        flush_other(&mut other, &mut tokens);
    }

    tokens.retain(|t| !t.is_empty() && !is_stopword(t));
    tokens.sort();
    tokens.dedup();
    tokens
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF
    )
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "the"
            | "and"
            | "for"
            | "with"
            | "this"
            | "that"
            | "from"
            | "you"
            | "are"
            | "\u{7684}"
            | "\u{4e86}"
            | "\u{548c}"
            | "\u{4e0e}"
            | "\u{5728}"
            | "\u{662f}"
            | "\u{6211}"
            | "\u{4f60}"
            | "\u{628a}"
            | "\u{8bf7}"
            | "\u{5e2e}"
    )
}

#[cfg(test)]
mod tests {
    use super::tokenize;

    #[test]
    fn splits_mixed_text_and_bigrams() {
        let tokens = tokenize("\u{5fae}\u{4fe1} \u{b7} \u{9879}\u{76ee}\u{7fa4} report.xlsx");
        assert!(tokens.contains(&"\u{5fae}\u{4fe1}".to_string()));
        assert!(tokens.contains(&"\u{9879}\u{76ee}".to_string()));
        assert!(tokens.contains(&"report".to_string()));
        assert!(tokens.contains(&"xlsx".to_string()));
    }

    #[test]
    fn drops_stopwords() {
        let tokens =
            tokenize("\u{5e2e}\u{6211}\u{6574}\u{7406}\u{4e00}\u{4e0b}\u{8fd9}\u{4e2a} file");
        assert!(!tokens.contains(&"\u{5e2e}".to_string()));
        assert!(tokens.contains(&"file".to_string()));
    }

    #[test]
    fn separates_cjk_from_digits_and_letters() {
        // CJK inside a mixed chunk must become its own token, otherwise it never matches.
        let tokens = tokenize(
            "\u{628a}\u{9879}\u{76ee}A 9\u{6708}\u{7535}\u{8d39}\u{6838}\u{5bf9}\u{4e00}\u{4e0b}",
        );
        assert!(
            tokens.contains(&"\u{7535}\u{8d39}".to_string()),
            "{tokens:?}"
        );
        assert!(
            tokens.contains(&"\u{6838}\u{5bf9}".to_string()),
            "{tokens:?}"
        );
        assert!(tokens.contains(&"9".to_string()), "{tokens:?}");
        assert!(tokens.contains(&"a".to_string()), "{tokens:?}");
    }

    #[test]
    fn matches_across_mixed_segments() {
        let a: std::collections::HashSet<String> = tokenize(
            "\u{628a}\u{9879}\u{76ee}A 9\u{6708}\u{7535}\u{8d39}\u{6838}\u{5bf9}\u{4e00}\u{4e0b}",
        )
        .into_iter()
        .collect();
        let b: std::collections::HashSet<String> =
            tokenize("\u{7535}\u{8d39}\u{6838}\u{5bf9} \u{b7} \u{9879}\u{76ee}A 9\u{6708}")
                .into_iter()
                .collect();
        assert!(!a.intersection(&b).collect::<Vec<_>>().is_empty());
    }
}
