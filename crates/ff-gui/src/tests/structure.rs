//! Tests for module `structure`: every role and group has plain-language text, and the English
//! labels stay free of Q/K/V, projection and MLP shorthand.

use super::*;
use crate::i18n::{I18n, Lang};

#[test]
fn every_role_has_a_human_title_and_description() {
    for role in Role::ALL {
        let (title, description) = role_keys(role);
        assert!(title.starts_with("structure-role-"), "{role:?}: {title}");
        assert!(description.starts_with("structure-role-"), "{role:?}: {description}");
        assert!(description.ends_with("-desc"), "{role:?}: {description}");
    }
}

#[test]
fn every_group_has_a_human_title_and_description() {
    for group in Group::ALL {
        let (title, description) = group_keys(group);
        assert!(title.starts_with("structure-group-"));
        assert!(description.ends_with("-desc"));
    }
}

#[test]
fn english_role_labels_avoid_technical_shorthand() {
    let t = I18n::new(Lang::En);
    let forbidden = ["query", "key", "value", "q/k/v", "qkv", "projection", "mlp", "dtype", "shape"];
    for role in Role::ALL {
        let (title, description) = role_keys(role);
        let text = format!("{} {}", t.t(title), t.t(description)).to_lowercase();
        for word in forbidden {
            assert!(!text.contains(word), "{role:?} exposes {word:?}: {text}");
        }
    }
    // The same ban covers the group headings of the Structure presentation.
    for group in Group::ALL {
        let (title, description) = group_keys(group);
        let text = format!("{} {}", t.t(title), t.t(description)).to_lowercase();
        for word in forbidden {
            assert!(!text.contains(word), "{group:?} exposes {word:?}: {text}");
        }
    }
}
