//! Domain model for FrankenForge: tensors, models, block layout and the ports
//! that format adapters implement. No I/O lives here.

pub mod decode;
pub mod dtype;
pub mod error;
pub mod layout;
pub mod model;
pub mod ports;
pub mod tensor;

pub use dtype::DType;
pub use error::{Error, Result};
pub use layout::{Block, Layout};
pub use model::{Model, selects};
pub use ports::{ModelFormat, TensorSource};
pub use tensor::TensorInfo;
