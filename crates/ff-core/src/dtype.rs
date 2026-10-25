use serde::{Deserialize, Serialize};

/// Element type, spelled as in the safetensors header.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DType {
    #[serde(rename = "BOOL")]
    Bool,
    U8,
    I8,
    #[serde(rename = "F8_E4M3")]
    F8E4M3,
    #[serde(rename = "F8_E5M2")]
    F8E5M2,
    U16,
    I16,
    F16,
    BF16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        use DType::*;
        match self {
            Bool | U8 | I8 | F8E4M3 | F8E5M2 => 1,
            U16 | I16 | F16 | BF16 => 2,
            U32 | I32 | F32 => 4,
            U64 | I64 | F64 => 8,
        }
    }
}
