//! Tests for module `lens`: table choice, nearest words over all or some columns, neuron words
//! for both storage orders, stopping, and refusing what cannot be read.

use std::collections::BTreeMap;

use ff_core::DType;

use super::*;

/// In-memory F32 model; each tensor is given as rows.
struct Rows(Model, BTreeMap<String, Vec<u8>>);

fn rows(tensors: &[(&str, Vec<Vec<f32>>)]) -> Rows {
    let mut m = Model::default();
    let mut bytes = BTreeMap::new();
    for (name, r) in tensors {
        m.insert(TensorInfo { name: (*name).into(), dtype: DType::F32, shape: vec![r.len(), r[0].len()] });
        bytes.insert((*name).to_owned(), r.iter().flatten().flat_map(|x| x.to_le_bytes()).collect());
    }
    Rows(m, bytes)
}

impl TensorSource for Rows {
    fn model(&self) -> &Model {
        &self.0
    }
    fn read(&self, name: &str) -> Result<Vec<u8>> {
        Ok(self.1[name].clone())
    }
    fn read_range(&self, name: &str, start: u64, len: u64) -> Result<Vec<u8>> {
        ff_core::ports::slice_range(&self.1[name], name, start, len)
    }
}

fn vocab() -> Vocab {
    Vocab::from_texts(["cat", "dog", "car", "sun"].map(String::from).to_vec())
}

/// cat and dog point one way, car another, sun a third.
fn embed() -> Vec<Vec<f32>> {
    vec![vec![1.0, 0.1, 0.0], vec![0.9, 0.2, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]]
}

fn run<T>(f: impl FnOnce(Watch) -> Result<Option<T>>) -> T {
    let (p, s) = (AtomicU64::new(0), AtomicBool::new(false));
    f(Watch { progress: &p, stop: &s }).unwrap().unwrap()
}

fn texts(h: &[Hit]) -> Vec<&str> {
    h.iter().map(|h| h.text.as_str()).collect()
}

#[test]
fn word_table_prefers_the_output_head_and_skips_per_layer_tables() {
    let src = rows(&[
        ("model.embed_tokens.weight", embed()),
        ("model.embed_tokens_per_layer.weight", embed()),
        ("lm_head.weight", embed()),
    ]);
    assert_eq!(word_table(src.model(), 4).as_deref(), Some("lm_head.weight"));
    let src =
        rows(&[("model.embed_tokens_per_layer.weight", embed()), ("model.embed_tokens.weight", embed())]);
    assert_eq!(word_table(src.model(), 4).as_deref(), Some("model.embed_tokens.weight"));
    assert_eq!(word_table(src.model(), 5), None, "fewer rows than tokens: not a word table");
}

#[test]
fn spelling_variants_of_one_word_count_once() {
    let v = Vocab::from_texts(["cat", " Cat", "CAT ", "dog", "car"].map(String::from).to_vec());
    let found = vec![(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6), (4, 0.5)];
    let h = hits(&v, found, 3);
    assert_eq!(texts(&h), ["cat", "dog", "car"], "the best-scored spelling stays");
}

#[test]
fn nearest_words_by_direction() {
    let src = rows(&[("e", embed())]);
    let hits = run(|w| nearest(&src, "e", &vocab(), 0, 0..3, 3, w));
    assert_eq!(texts(&hits), ["cat", "dog", "car"]);
    assert!((hits[0].score - 1.0).abs() < 1e-6);
    // Only the last column: sun is the one non-zero row, and cat's slice is zero.
    assert_eq!(run(|w| nearest(&src, "e", &vocab(), 3, 2..3, 2, w))[0].text, "sun");
    assert!(
        run(|w| nearest(&src, "e", &vocab(), 0, 2..3, 2, w)).is_empty(),
        "zero slice: nothing to compare"
    );
    assert!(
        nearest(
            &src,
            "e",
            &vocab(),
            9,
            0..3,
            2,
            Watch { progress: &AtomicU64::new(0), stop: &AtomicBool::new(false) }
        )
        .is_err()
    );
}

#[test]
fn neurons_read_in_both_storage_orders() {
    // What each neuron writes: 0 strongly toward cat, 1 weakly toward car, 2 toward sun.
    let writes = [[3.0, 0.5, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]];
    // Usual `[width, inner]` storage: neuron j is column j.
    let columns: Vec<Vec<f32>> = (0..3).map(|i| writes.iter().map(|n| n[i]).collect()).collect();
    let src = rows(&[("e", embed()), ("down", columns.clone())]);
    let found = run(|w| neurons(&src, "down", "e", &vocab(), 2, 1, w));
    assert_eq!(found.iter().map(|n| n.neuron).collect::<Vec<_>>(), [0, 2]);
    assert_eq!(found[0].words[0].text, "cat");
    assert_eq!(found[1].words[0].text, "sun");

    // The same neurons stored [inner, width] (GPT-2 style): neuron j is row j.
    let transposed: Vec<Vec<f32>> = writes.iter().map(|n| n.to_vec()).collect();
    let src = rows(&[("e", embed()), ("down", transposed)]);
    let found = run(|w| neurons(&src, "down", "e", &vocab(), 2, 1, w));
    assert_eq!((found[0].neuron, found[0].words[0].text.as_str()), (0, "cat"));
}

#[test]
fn stopping_and_mismatched_matrices() {
    let src = rows(&[("e", embed()), ("down", vec![vec![1.0, 2.0]; 2])]);
    let (p, s) = (AtomicU64::new(0), AtomicBool::new(true));
    let w = Watch { progress: &p, stop: &s };
    assert_eq!(nearest(&src, "e", &vocab(), 0, 0..3, 2, w).unwrap(), None);
    let go = AtomicBool::new(false);
    let w = Watch { progress: &p, stop: &go };
    assert!(neurons(&src, "down", "e", &vocab(), 1, 1, w).is_err(), "2x2 does not match width 3");
    assert_eq!(neurons_bytes(src.model(), "down", "e"), 2 * 16 + 48);
}
