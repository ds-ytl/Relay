//! Opaque handles for applications (`app_12`) and elements (`el_71`).
//!
//! The model never sees a platform pointer: no `AXUIElement` address, no COM
//! pointer, no D-Bus object path reaches the wire. `op=apps`, `op=query`,
//! `op=tree`, and `op=hit_test` hand out handles; `op=inspect`, `op=action`,
//! and `op=wait` take them back, and this registry translates a handle into
//! the [`ElementData`] snapshot the platform backend produced.
//!
//! Handles are deliberately **short-lived**. A React re-render, an Electron
//! reload, or a recreated window invalidates the underlying node, so an
//! element handle is a hint, not an identity: `op=wait` re-resolves it against
//! the live tree, and a handle for an element that is gone (or was trimmed out
//! of a full registry) answers `stale_element` — which tells the model to
//! `query` again instead of retrying the same call.

use std::collections::HashMap;

use xa11y::ElementData;

use crate::envelope::{Failure, codes};

/// Element handles kept alive at once. Reaching the cap drops every element
/// handle (applications stay) — the plan's "handles are short-lived" contract
/// applied to a long-running server.
pub const MAX_TRACKED_ELEMENTS: usize = 4096;

/// Application handles kept alive at once.
pub const MAX_TRACKED_APPS: usize = 64;

/// Handle bookkeeping for one server process.
#[derive(Debug, Default)]
pub struct Registry {
    apps: HashMap<String, ElementData>,
    elements: HashMap<String, ElementData>,
    app_handles: HashMap<String, String>,
    element_handles: HashMap<String, String>,
    next_id: u64,
}

impl Registry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of element handles currently resolvable.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Interns one application snapshot, reusing its handle when the same
    /// application was seen before.
    pub fn intern_app(&mut self, data: &ElementData) -> String {
        if self.apps.len() >= MAX_TRACKED_APPS
            && !self.app_handles.contains_key(&Self::identity(data))
        {
            // Applications are few and stable; a full table means the
            // enumerations are churning, so starting over is cheaper than
            // evicting arbitrary entries and keeping dead handles alive.
            self.apps.clear();
            self.app_handles.clear();
        }
        intern(
            &mut self.apps,
            &mut self.app_handles,
            &mut self.next_id,
            "app",
            data,
        )
    }

    /// Interns one element snapshot, reusing its handle when the same element
    /// was seen before.
    pub fn intern_element(&mut self, data: &ElementData) -> String {
        if self.elements.len() >= MAX_TRACKED_ELEMENTS
            && !self.element_handles.contains_key(&Self::identity(data))
        {
            self.elements.clear();
            self.element_handles.clear();
        }
        intern(
            &mut self.elements,
            &mut self.element_handles,
            &mut self.next_id,
            "el",
            data,
        )
    }

    /// Resolves an application handle.
    ///
    /// # Errors
    ///
    /// Returns `app_not_found` for a handle this registry never issued or has
    /// already dropped.
    pub fn app(&self, handle: &str) -> Result<ElementData, Failure> {
        self.apps.get(handle).cloned().ok_or_else(|| {
            Failure::new(
                codes::APP_NOT_FOUND,
                format!("unknown application handle {handle:?}; call op=apps for a fresh one"),
            )
        })
    }

    /// Resolves an element handle.
    ///
    /// # Errors
    ///
    /// Returns `stale_element` for a handle this registry never issued or has
    /// already dropped, which is the model's signal to `query` again.
    pub fn element(&self, handle: &str) -> Result<ElementData, Failure> {
        self.elements.get(handle).cloned().ok_or_else(|| {
            Failure::new(
                codes::STALE_ELEMENT,
                format!("element handle {handle:?} is no longer valid; query the element again"),
            )
        })
    }

    /// Platform identity of one snapshot.
    ///
    /// Keyed on `stable_id` (macOS `AXIdentifier`, Windows `hwnd:…` /
    /// `AutomationId`, Linux D-Bus object path — scoped by bus name, since one
    /// process can expose distinct windows under one path) and falling back to
    /// the provider-assigned handle, mirroring `xa11y`'s own window identity
    /// rule. Presentation data (title, bounds) is never used as identity: two
    /// windows can share both.
    #[must_use]
    pub fn identity(data: &ElementData) -> String {
        data.stable_id.clone().map_or_else(
            || {
                let pid = data
                    .pid
                    .map_or_else(|| "-".to_string(), |pid| pid.to_string());
                format!("{pid}:h{}", data.handle)
            },
            |stable| match data.raw.get("bus_name") {
                Some(serde_json::Value::String(bus)) => format!("{bus}:{stable}"),
                _ => stable,
            },
        )
    }
}

/// Shared body of both interning paths: reuse the handle of a known identity,
/// otherwise mint one and remember the snapshot.
fn intern(
    store: &mut HashMap<String, ElementData>,
    handles: &mut HashMap<String, String>,
    next_id: &mut u64,
    prefix: &str,
    data: &ElementData,
) -> String {
    let key = Registry::identity(data);
    if let Some(handle) = handles.get(&key) {
        store.insert(handle.clone(), data.clone());
        return handle.clone();
    }
    *next_id += 1;
    let handle = format!("{prefix}_{}", *next_id);
    handles.insert(key, handle.clone());
    store.insert(handle.clone(), data.clone());
    handle
}

#[cfg(test)]
mod tests {
    use super::*;
    use xa11y::{Rect, Role};

    fn element(name: &str, stable: Option<&str>, handle: u64) -> ElementData {
        let mut data = ElementData::for_role(Role::Button);
        data.name = Some(name.to_string());
        data.handle = handle;
        data.stable_id = stable.map(str::to_string);
        data.bounds = Some(Rect {
            x: 0,
            y: 0,
            width: 10,
            height: 10,
        });
        data
    }

    #[test]
    fn the_same_identity_keeps_its_handle_across_enumerations() {
        let mut registry = Registry::new();
        let first = registry.intern_element(&element("Back", Some("btn-back"), 3));
        let second = registry.intern_element(&element("Back", Some("btn-back"), 3));
        assert_eq!(first, second);
        assert_eq!(registry.element_count(), 1);
    }

    #[test]
    fn stable_ids_are_scoped_by_bus_name() {
        let mut first = element("Window", Some("/org/a11y/win/1"), 0);
        first
            .raw
            .insert("bus_name".into(), serde_json::json!(":1.7"));
        let mut second = element("Window", Some("/org/a11y/win/1"), 0);
        second
            .raw
            .insert("bus_name".into(), serde_json::json!(":1.8"));
        assert_ne!(Registry::identity(&first), Registry::identity(&second));
    }

    #[test]
    fn elements_without_a_stable_id_fall_back_to_pid_and_handle() {
        assert_eq!(Registry::identity(&element("Back", None, 7)), "-:h7");
    }

    #[test]
    fn unknown_handles_are_stale_or_missing_not_silent_misses() {
        let registry = Registry::new();
        let stale = registry.element("el_9").unwrap_err();
        assert_eq!(stale.code, codes::STALE_ELEMENT);
        assert!(stale.retryable);
        let missing = registry.app("app_9").unwrap_err();
        assert_eq!(missing.code, codes::APP_NOT_FOUND);
    }

    #[test]
    fn trimming_drops_element_handles_so_they_read_as_stale() {
        let mut registry = Registry::new();
        let first = registry.intern_element(&element("Back", Some("btn-back"), 1));
        for index in 0..MAX_TRACKED_ELEMENTS {
            registry.intern_element(&element(
                &format!("row {index}"),
                Some(&index.to_string()),
                index as u64 + 10,
            ));
        }
        assert!(registry.element(&first).is_err());
        assert_eq!(registry.element_count(), 1);
    }
}
