//! Tests for module `delta`: known arrays in every supported dtype, zero base norm, unsupported
//! dtypes, bounded chunked reads, and block aggregates through `Report::measure`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use ff_core::{DType, Model, TensorInfo};

use super::*;
use crate::Step;

/// In-memory source with real bytes; remembers the largest read.
#[derive(Default)]
struct Values {
    model: Model,
    bytes: BTreeMap<String, Vec<u8>>,
    largest_read: AtomicU64,
}

impl Values {
    fn with(mut self, name: &str, dtype: DType, bytes: Vec<u8>) -> Self {
        let shape = vec![bytes.len() / dtype.size_bytes()];
        self.model.insert(TensorInfo { name: name.into(), dtype, shape });
        self.bytes.insert(name.into(), bytes);
        self
    }
}

impl TensorSource for Values {
    fn model(&self) -> &Model {
        &self.model
    }

    fn read(&self, name: &str) -> Result<Vec<u8>> {
        Ok(self.bytes[name].clone())
    }

    fn read_range(&self, name: &str, start: u64, len: u64) -> Result<Vec<u8>> {
        self.largest_read.fetch_max(len, Ordering::Relaxed);
        ff_core::ports::slice_range(&self.bytes[name], name, start, len)
    }
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn u16s(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn delta(dtype: DType, new: Vec<u8>, old: Vec<u8>) -> Delta {
    let (n, o) = (Values::default().with("t", dtype, new), Values::default().with("t", dtype, old));
    sums(&n, "t", &o, "t").unwrap().map_or_else(|r| unknown(&r), Sums::relative)
}

fn value(d: Delta) -> f64 {
    let Delta::Relative { value } = d else { panic!("{d:?}") };
    value
}

#[test]
fn f32_known_arrays() {
    // old = (3, 4), new = (3, 0): ||diff|| = 4, ||old|| = 5.
    assert!((value(delta(DType::F32, f32s(&[3.0, 0.0]), f32s(&[3.0, 4.0]))) - 0.8).abs() < 1e-12);
    assert_eq!(value(delta(DType::F32, f32s(&[1.0, 2.0]), f32s(&[1.0, 2.0]))), 0.0);
}

#[test]
fn f16_and_bf16_known_arrays() {
    // old = (1, -2), new = (1, 0): sqrt(4 / 5).
    let want = (4.0f64 / 5.0).sqrt();
    let f16 = delta(DType::F16, u16s(&[0x3C00, 0x0000]), u16s(&[0x3C00, 0xC000]));
    assert!((value(f16) - want).abs() < 1e-12);
    let bf16 = delta(DType::BF16, u16s(&[0x3F80, 0x0000]), u16s(&[0x3F80, 0xC000]));
    assert!((value(bf16) - want).abs() < 1e-12);
}

#[test]
fn zero_base_unsupported_dtype_and_nan_are_unknown_not_zero() {
    let zero = delta(DType::F32, f32s(&[1.0, 2.0]), f32s(&[0.0, 0.0]));
    assert!(matches!(&zero, Delta::Unknown { reason } if reason.contains("zeros")), "{zero:?}");
    let int = delta(DType::I32, vec![1; 8], vec![2; 8]);
    assert!(matches!(&int, Delta::Unknown { reason } if reason.contains("I32")), "{int:?}");
    let nan = delta(DType::F32, f32s(&[f32::NAN]), f32s(&[1.0]));
    assert!(matches!(nan, Delta::Unknown { .. }), "{nan:?}");
}

#[test]
fn layout_mismatch_is_unknown() {
    let n = Values::default().with("t", DType::F32, f32s(&[1.0, 2.0]));
    let o = Values::default().with("t", DType::F16, u16s(&[0x3C00, 0x3C00, 0x3C00, 0x3C00]));
    assert!(sums(&n, "t", &o, "t").unwrap().is_err());
}

#[test]
fn large_tensor_is_streamed_in_bounded_chunks() {
    let count = (CHUNK as usize / 4) * 2 + 3;
    let n = Values::default().with("t", DType::F32, f32s(&vec![2.0; count]));
    let o = Values::default().with("t", DType::F32, f32s(&vec![1.0; count]));
    let s = sums(&n, "t", &o, "t").unwrap().unwrap();
    assert_eq!(s.relative(), Delta::Relative { value: 1.0 });
    assert_eq!(n.largest_read.load(Ordering::Relaxed), CHUNK);
    assert_eq!(o.largest_read.load(Ordering::Relaxed), CHUNK);
}

#[test]
fn measure_fills_tensor_and_block_deltas() {
    let base = Values::default()
        .with("m.layers.0.w", DType::F32, f32s(&[3.0, 4.0]))
        .with("m.layers.0.b", DType::F32, f32s(&[0.0, 0.0]))
        .with("head", DType::F32, f32s(&[1.0]));
    let donor = Values::default()
        .with("m.layers.0.w", DType::F32, f32s(&[3.0, 0.0]))
        .with("m.layers.0.b", DType::F32, f32s(&[0.0, 3.0]))
        .with("head", DType::F32, f32s(&[1.0]));
    let map: BTreeMap<String, Box<dyn TensorSource>> = BTreeMap::from([
        ("a".to_owned(), Box::new(base) as Box<dyn TensorSource>),
        ("b".to_owned(), Box::new(donor) as _),
    ]);
    let sources = Sources::new(map);
    let mut plan = Plan::from_base("a", sources.get("a").unwrap().model());
    let step = Step::Transplant {
        from: "b".into(),
        src: "m.layers.0.".into(),
        dst: "m.layers.0.".into(),
        keep_extra: false,
    };
    crate::ops::apply(&mut plan, &step, &sources).unwrap();
    let mut r = Report::new("a", sources.get("a").unwrap().model(), &plan);
    r.measure(&plan, &sources).unwrap();

    let block = &r.stacks[0].blocks[0];
    let by_name: BTreeMap<&str, &Delta> =
        block.replaced.iter().map(|t| (t.name.as_str(), &t.delta)).collect();
    assert!(matches!(by_name["m.layers.0.b"], Delta::Unknown { .. }), "zero base tensor");
    assert!((value(by_name["m.layers.0.w"].clone()) - 0.8).abs() < 1e-12);
    // The block sums squares over its tensors: (16 + 9) / (25 + 0), so a zero tensor still counts.
    assert_eq!(block.delta, Some(Delta::Relative { value: 1.0 }));
    assert!(r.outside[0].delta.is_none(), "head was not replaced");
}

#[test]
fn block_aggregate_sums_squares_over_tensors() {
    let a = Sums { diff_sq: 16.0, old_sq: 25.0 };
    let b = Sums { diff_sq: 9.0, old_sq: 0.0 };
    assert_eq!((a + b).relative(), Delta::Relative { value: 1.0 });
}
