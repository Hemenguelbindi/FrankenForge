//! Tests for module `safetensors`.

use super::*;
use ff_core::DType;

#[test]
fn roundtrip() {
    let dir = std::env::temp_dir().join(format!("ff-io-roundtrip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.safetensors");

    let mut model = Model::default();
    model.metadata.insert("format".into(), "pt".into());
    model.insert(TensorInfo { name: "a".into(), dtype: DType::F32, shape: vec![2] });
    model.insert(TensorInfo { name: "b".into(), dtype: DType::U8, shape: vec![3] });
    Safetensors.write(&path, &model, &mut |t| Ok((0..t.byte_len() as u8).collect())).unwrap();

    let back = Safetensors.open(&path).unwrap();
    assert_eq!(back.model().tensors, model.tensors);
    assert_eq!(back.model().metadata, model.metadata);
    assert_eq!(back.read("b").unwrap(), vec![0, 1, 2]);
    // A directory with one shard opens as the same model.
    assert_eq!(Safetensors.open(&dir).unwrap().read("a").unwrap().len(), 8);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_range_streams_part_of_a_tensor_and_rejects_bad_ranges() {
    let dir = std::env::temp_dir().join(format!("ff-io-range-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.safetensors");
    let mut model = Model::default();
    model.insert(TensorInfo { name: "a".into(), dtype: DType::U8, shape: vec![4] });
    model.insert(TensorInfo { name: "b".into(), dtype: DType::U8, shape: vec![6] });
    Safetensors.write(&path, &model, &mut |t| Ok((10..10 + t.byte_len() as u8).collect())).unwrap();

    for source in [Safetensors.open(&path).unwrap(), Safetensors.open(&dir).unwrap()] {
        assert_eq!(source.read_range("b", 2, 3).unwrap(), [12, 13, 14]);
        assert_eq!(source.read_range("b", 6, 0).unwrap(), Vec::<u8>::new());
        // Past the end of this tensor (even though "a" lies before it in the file).
        assert!(source.read_range("b", 4, 3).is_err());
        assert!(source.read_range("b", u64::MAX, 2).is_err(), "overflowing range");
        assert!(source.read_range("missing", 0, 1).is_err());
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
