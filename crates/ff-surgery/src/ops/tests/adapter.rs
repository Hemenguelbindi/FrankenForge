//! Tests for module `adapter`.

use super::*;

#[test]
fn identity_on_overlap() {
    let info = TensorInfo { name: "p".into(), dtype: DType::F32, shape: vec![2, 3] };
    let floats: Vec<f32> = materialize(&info, AdapterInit::Identity)
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(floats, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
}
