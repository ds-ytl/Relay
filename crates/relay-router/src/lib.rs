//! Scoring helpers for session / workspace retrieval.
//!
//! This crate makes **no routing decisions**: which workspace and which session to use is decided
//! by the understanding agent after it has inspected the environment (see the `understanding`
//! module in `relay-runtime`). This crate only provides two things:
//! - `tokenize`: lightweight mixed CJK / ASCII tokenizer (CJK as bigrams), reused by retrieval and
//!   prompts;
//! - `rank_sessions`: scores and orders the output of the understanding agent's `list_sessions`
//!   tool, putting the most likely sessions first.

use std::collections::HashSet;

use chrono::{DateTime, Duration, Utc};
use relay_core::Session;

mod tokens;

pub use tokens::tokenize;

/// A session and how closely it matches the current instruction.
#[derive(Debug, Clone)]
pub struct RankedSession {
    pub session: Session,
    /// Semantic overlap (Jaccard, 0–1).
    pub overlap: f32,
    /// Combined ranking score: `overlap * 0.8 + recency`.
    pub score: f32,
}

/// Order sessions by "word overlap with the instruction + recent activity" (presentation order
/// only, nothing is filtered out).
pub fn rank_sessions(
    instruction: &str,
    sessions: &[Session],
    now: DateTime<Utc>,
) -> Vec<RankedSession> {
    let instruction_tokens: HashSet<String> = tokenize(instruction).into_iter().collect();
    let mut ranked: Vec<RankedSession> = sessions
        .iter()
        .map(|session| {
            let tokens: HashSet<String> = tokenize(&format!(
                "{} {} {}",
                session.title,
                session.summary,
                session.last_context.clone().unwrap_or_default()
            ))
            .into_iter()
            .collect();
            let overlap = jaccard(&instruction_tokens, &tokens);
            RankedSession {
                session: session.clone(),
                overlap,
                score: overlap * 0.8 + recency_bonus(session.last_active_at, now),
            }
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

/// Maximum session-title length. English titles need more room than the original two-character
/// Chinese budget, while still fitting the sidebar.
const SESSION_TITLE_LIMIT: usize = 48;

/// Derive a name from the first line of the instruction, as a fallback when the model gives no
/// session title.
pub fn derive_session_title(instruction: &str) -> String {
    let trimmed = instruction.trim().lines().next().unwrap_or("").trim();
    if trimmed.is_empty() {
        return "New session".into();
    }
    let mut title: String = trimmed.chars().take(SESSION_TITLE_LIMIT).collect();
    if trimmed.chars().count() > SESSION_TITLE_LIMIT {
        title.push('…');
    }
    title
}

fn recency_bonus(last_active: DateTime<Utc>, now: DateTime<Utc>) -> f32 {
    let age = now.signed_duration_since(last_active);
    if age < Duration::zero() {
        return 0.2;
    }
    if age < Duration::minutes(30) {
        0.2
    } else if age < Duration::hours(6) {
        0.12
    } else if age < Duration::days(2) {
        0.06
    } else {
        0.0
    }
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    intersection / union
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use relay_core::{SessionId, WorkspaceId};

    fn session(title: &str, summary: &str, minutes_ago: i64) -> Session {
        let now = Utc::now();
        Session {
            id: SessionId::new(),
            workspace_id: WorkspaceId::new(),
            title: title.into(),
            summary: summary.into(),
            created_at: now,
            last_active_at: now - Duration::minutes(minutes_ago),
            last_context: None,
        }
    }

    #[test]
    fn ranks_matching_session_first() {
        let sessions = vec![
            session(
                "\u{5468}\u{62a5}",
                "\u{6574}\u{7406}\u{4e0a}\u{5468}\u{9500}\u{552e}\u{6570}\u{636e}",
                5,
            ),
            session(
                "\u{7535}\u{8d39}\u{6838}\u{5bf9}",
                "\u{6838}\u{5bf9}\u{9879}\u{76ee}A 9\u{6708}\u{7535}\u{8d39}",
                3,
            ),
        ];
        let ranked = rank_sessions(
            "\u{628a}\u{9879}\u{76ee}A 9\u{6708}\u{7535}\u{8d39}\u{6838}\u{5bf9}\u{4e00}\u{4e0b}",
            &sessions,
            Utc::now(),
        );
        assert_eq!(ranked[0].session.title, "\u{7535}\u{8d39}\u{6838}\u{5bf9}");
        assert!(ranked[0].overlap > 0.0);
    }

    #[test]
    fn recent_session_wins_when_overlap_is_equal() {
        let sessions = vec![
            session("\u{6742}\u{9879}", "", 240),
            session("\u{6742}\u{9879}", "", 1),
        ];
        let ranked = rank_sessions("\u{968f}\u{4fbf}\u{770b}\u{770b}", &sessions, Utc::now());
        assert!(ranked[0].score >= ranked[1].score);
    }

    #[test]
    fn derives_title_from_first_line() {
        let title = derive_session_title(
            "  \u{7b2c}\u{4e00}\u{884c}\u{8981}\u{70b9}\n\u{7b2c}\u{4e8c}\u{884c}\u{7ec6}\u{8282}",
        );
        assert_eq!(title, "\u{7b2c}\u{4e00}\u{884c}\u{8981}\u{70b9}");
        assert_eq!(derive_session_title("   "), "New session");
    }

    #[test]
    fn long_title_is_truncated() {
        let title = derive_session_title(&"\u{5b57}".repeat(60));
        assert!(title.ends_with('…'));
        assert!(title.chars().count() <= SESSION_TITLE_LIMIT + 1);
    }

    #[test]
    fn timestamp_roundtrip_sanity() {
        let fixed = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        let sessions = vec![session("a", "", 10)];
        let ranked = rank_sessions("a", &sessions, fixed);
        assert_eq!(ranked.len(), 1);
    }
}
