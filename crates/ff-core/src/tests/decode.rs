//! Tests for module `decode`.

use super::*;

#[test]
fn f16_decodes_exactly() {
    assert_eq!(f16_to_f32(0x3C00), 1.0);
    assert_eq!(f16_to_f32(0xC000), -2.0);
    assert_eq!(f16_to_f32(0x3800), 0.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0, "largest finite half");
    assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24), "smallest subnormal");
    assert_eq!(f16_to_f32(0x8000), 0.0);
    assert!(f16_to_f32(0x8000).is_sign_negative());
    assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
    assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
    assert!(f16_to_f32(0x7E00).is_nan());
}

#[test]
fn decoder_covers_float_dtypes_only() {
    let (size, d) = decoder(DType::BF16).unwrap();
    assert_eq!((size, d(&[0x80, 0x3F])), (2, 1.0));
    let (size, d) = decoder(DType::F16).unwrap();
    assert_eq!((size, d(&[0x00, 0xC0])), (2, -2.0));
    let (size, d) = decoder(DType::F32).unwrap();
    assert_eq!((size, d(&0.5f32.to_le_bytes())), (4, 0.5));
    for other in [DType::F64, DType::I32, DType::U8, DType::F8E4M3] {
        assert!(decoder(other).is_none(), "{other:?}");
    }
}
