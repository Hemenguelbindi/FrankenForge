//! Tests for module `weights`: which blocks are compared, known per-block deltas, stopping, text.

use std::collections::BTreeMap;

use ff_core::{DType, TensorInfo};

use super::*;

/// In-memory F32 model; every tensor is a vector of the given values.
struct Values(Model, BTreeMap<String, Vec<u8>>);

fn values(tensors: &[(&str, &[f32])]) -> Values {
    let mut m = Model::default();
    let mut bytes = BTreeMap::new();
    for (name, v) in tensors {
        m.insert(TensorInfo { name: (*name).into(), dtype: DType::F32, shape: vec![v.len()] });
        bytes.insert((*name).to_owned(), v.iter().flat_map(|x| x.to_le_bytes()).collect());
    }
    Values(m, bytes)
}

impl TensorSource for Values {
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

fn run(base: &Values, donor: &Values) -> WeightDiff {
    diff(base, donor, &AtomicU64::new(0), &AtomicBool::new(false)).unwrap().unwrap()
}

#[test]
fn per_block_delta_and_most_changed_tensor() {
    let base = values(&[
        ("m.layers.0.a", &[3.0, 4.0]),
        ("m.layers.0.b", &[1.0, 0.0]),
        ("m.layers.1.a", &[1.0, 1.0]),
        ("head", &[1.0]),
    ]);
    let donor = values(&[
        ("m.layers.0.a", &[3.0, 4.0]),
        ("m.layers.0.b", &[1.0, 1.0]),
        ("m.layers.1.a", &[1.0, 1.0]),
        ("head", &[5.0]),
    ]);
    let d = run(&base, &donor);
    assert_eq!(d.not_compared, 0);
    assert_eq!(d.blocks.len(), 2, "tensors outside blocks are not part of the block map");
    // Block 0: diff^2 = 1, base^2 = 25 + 1.
    assert_eq!(d.blocks[0].delta, Delta::Relative { value: (1.0f64 / 26.0).sqrt() });
    assert_eq!(d.blocks[0].most_changed(), Some(("m.layers.0.b", 1.0)));
    assert_eq!(d.blocks[0].params, 4);
    assert_eq!(d.blocks[1].delta, Delta::Relative { value: 0.0 });
}

#[test]
fn blocks_with_other_structure_are_counted_not_compared() {
    let base = values(&[("m.layers.0.a", &[1.0]), ("m.layers.1.a", &[1.0]), ("m.layers.2.a", &[1.0])]);
    let donor = values(&[
        ("m.layers.0.a", &[1.0, 2.0]),
        ("m.layers.1.a", &[1.0]),
        ("m.layers.1.extra", &[1.0]),
        ("m.layers.2.a", &[2.0]),
    ]);
    let d = run(&base, &donor);
    assert_eq!(d.blocks.iter().map(|b| b.index).collect::<Vec<_>>(), [2]);
    assert_eq!(d.not_compared, 2);
    assert_eq!(bytes_to_read(base.model(), donor.model()), 8);
}

#[test]
fn unrelated_models_have_nothing_to_compare() {
    let d = run(&values(&[("m.layers.0.a", &[1.0])]), &values(&[("x.h.0.w", &[1.0])]));
    assert!(d.blocks.is_empty());
    assert!(d.text().contains("no block has the same tensors"), "{}", d.text());
}

#[test]
fn stop_returns_none() {
    let m = values(&[("m.layers.0.a", &[1.0])]);
    assert_eq!(diff(&m, &m, &AtomicU64::new(0), &AtomicBool::new(true)).unwrap(), None);
}

#[test]
fn text_shows_bars_and_short_names() {
    let base = values(&[("m.layers.0.mlp.w", &[1.0]), ("m.layers.1.mlp.w", &[1.0])]);
    let donor = values(&[("m.layers.0.mlp.w", &[1.5]), ("m.layers.1.mlp.w", &[1.25])]);
    let text = run(&base, &donor).text();
    assert!(text.contains("    0  ####################    50.00%   most changed: mlp.w 50.00%"), "{text}");
    assert!(text.contains("    1  ##########              25.00%"), "{text}");
    assert_eq!(short("model.layers.15.mlp.up_proj.weight", "model.layers"), "mlp.up_proj.weight");
}
