//! Groups an opened checkpoint into components (language, audio, vision, other) using only the
//! names actually present in the file: a modality is shown only when a name segment says so.

use std::collections::BTreeMap;

use ff_core::{Layout, Model};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Kind {
    Language,
    Audio,
    Vision,
    Other,
}

impl Kind {
    #[cfg(test)]
    pub const ALL: [Kind; 4] = [Kind::Language, Kind::Audio, Kind::Vision, Kind::Other];

    /// Message keys in `locales/*.ftl`: (title, plain-language purpose).
    pub fn message_keys(self) -> (&'static str, &'static str) {
        match self {
            Kind::Language => ("component-language", "component-language-purpose"),
            Kind::Audio => ("component-audio", "component-audio-purpose"),
            Kind::Vision => ("component-vision", "component-vision-purpose"),
            Kind::Other => ("component-other", "component-other-purpose"),
        }
    }
}

/// One stack of repeated blocks; `blocks` index into `Layout::blocks`.
#[derive(Debug, PartialEq, Eq)]
pub struct Stack {
    pub name: String,
    pub blocks: Vec<usize>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Component {
    pub kind: Kind,
    pub stacks: Vec<Stack>,
    /// Tensors outside any block that belong to this component.
    pub loose: Vec<String>,
    pub params: usize,
}

impl Component {
    pub fn block_count(&self) -> usize {
        self.stacks.iter().map(|s| s.blocks.len()).sum()
    }
}

/// Audio and vision are recognised only by a whole word in the name (`audio_tower`,
/// `embed_vision`, `visual`), so `revision` is not mistaken for vision.
pub(crate) fn modality(name: &str) -> Option<Kind> {
    for word in name.split(['.', '_']) {
        match word {
            "audio" => return Some(Kind::Audio),
            "vision" | "visual" => return Some(Kind::Vision),
            _ => {}
        }
    }
    None
}

/// Token-embedding tables: their presence next to a stack is what makes it a language stack.
const TOKEN_EMBEDDINGS: &[&str] = &["embed_tokens", "wte", "word_embeddings"];

/// `model.language_model.layers` -> `model.language_model.`
fn root_of(stack: &str) -> Option<String> {
    stack.rsplit_once('.').map(|(parent, _)| format!("{parent}."))
}

/// Language needs evidence too: the word "language"/"text" in the stack name, or a
/// token-embedding table under the same parent (`model.embed_tokens` next to `model.layers`).
fn is_language(stack: &str, model: &Model) -> bool {
    if stack.split(['.', '_']).any(|w| w == "language" || w == "text") {
        return true;
    }
    let Some(root) = root_of(stack) else { return false };
    model.tensors.keys().any(|n| {
        n.strip_prefix(root.as_str())
            .is_some_and(|rest| rest.split('.').any(|w| TOKEN_EMBEDDINGS.contains(&w)))
    })
}

/// Components in `Kind` order; kinds with nothing found are left out.
pub fn components(model: &Model, layout: &Layout) -> Vec<Component> {
    let mut found: BTreeMap<Kind, Component> = BTreeMap::new();
    fn entry(found: &mut BTreeMap<Kind, Component>, kind: Kind) -> &mut Component {
        found.entry(kind).or_insert_with(|| Component {
            kind,
            stacks: Vec::new(),
            loose: Vec::new(),
            params: 0,
        })
    }

    // Blocks are sorted by stack, so each stack is classified once, when it starts.
    let mut language_roots: Vec<String> = Vec::new();
    let mut current: Option<(&str, Kind)> = None;
    for (i, block) in layout.blocks.iter().enumerate() {
        let kind = match current {
            Some((stack, kind)) if stack == block.stack => kind,
            _ => {
                let kind = modality(&block.stack)
                    .or_else(|| is_language(&block.stack, model).then_some(Kind::Language))
                    .unwrap_or(Kind::Other);
                if kind == Kind::Language
                    && let Some(root) = root_of(&block.stack)
                    && !language_roots.contains(&root)
                {
                    language_roots.push(root);
                }
                current = Some((&block.stack, kind));
                kind
            }
        };
        let c = entry(&mut found, kind);
        match c.stacks.last_mut() {
            Some(s) if s.name == block.stack => s.blocks.push(i),
            _ => c.stacks.push(Stack { name: block.stack.clone(), blocks: vec![i] }),
        }
        c.params += block.tensors.iter().map(|n| model.tensors[n].numel()).sum::<usize>();
    }

    // Loose tensors go with a modality named in them, else with the language part they sit under
    // (`model.language_model.embed_tokens` next to `model.language_model.layers`), else Other.
    for name in &layout.other {
        let kind = modality(name)
            .or_else(|| language_roots.iter().any(|r| name.starts_with(r.as_str())).then_some(Kind::Language))
            .unwrap_or(Kind::Other);
        let c = entry(&mut found, kind);
        c.loose.push(name.clone());
        c.params += model.tensors[name].numel();
    }
    found.into_values().collect()
}

#[cfg(test)]
#[path = "tests/components.rs"]
mod tests;
