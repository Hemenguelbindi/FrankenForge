use crate::DType;

/// Metadata of one tensor; the bytes stay in the source until needed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// `TensorInfo` comes from a validated header, so this cannot overflow.
    pub fn byte_len(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }

    pub fn same_layout(&self, other: &TensorInfo) -> bool {
        self.dtype == other.dtype && self.shape == other.shape
    }
}
