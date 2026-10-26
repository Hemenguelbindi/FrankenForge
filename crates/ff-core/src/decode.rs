//! Little-endian float decoding shared by everything that looks at weight values (the GUI's
//! tensor profile, the build report's weight delta). Pure functions; the bytes come from a port.

use crate::DType;

pub type Decode = fn(&[u8]) -> f64;

/// Element size and decoder for the float dtypes whose values we look at. F64 is left out on
/// purpose: its extremes would overflow f64 accumulators, which are exact for 16/32-bit input.
pub fn decoder(dtype: DType) -> Option<(usize, Decode)> {
    let d: (usize, Decode) = match dtype {
        DType::F32 => (4, |b| f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))),
        DType::BF16 => (2, |b| f64::from(f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))),
        DType::F16 => (2, |b| f64::from(f16_to_f32(u16::from_le_bytes([b[0], b[1]])))),
        _ => return None,
    };
    Some(d)
}

/// IEEE 754 half precision to single precision, exact for every input.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let man = u32::from(h & 0x03ff);
    match exp {
        // Subnormal: man * 2^-24, exact because man < 2^10.
        0 => sign * man as f32 * (1.0 / 16_777_216.0),
        0x1f if man == 0 => sign * f32::INFINITY,
        0x1f => f32::NAN,
        _ => f32::from_bits((u32::from(h & 0x8000) << 16) | ((u32::from(exp) + 112) << 23) | (man << 13)),
    }
}

#[cfg(test)]
#[path = "tests/decode.rs"]
mod tests;
