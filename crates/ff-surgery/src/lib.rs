//! Use cases: inspect models, turn a recipe into a plan, build and verify
//! the result. Depends only on `ff-core` ports, never on a concrete format.

pub mod build;
pub mod companion;
pub mod compare;
pub mod inspect;
pub mod lock;
pub mod ops;
pub mod plan;
pub mod recipe;
pub mod sources;

pub use plan::{Experimental, Payload, Plan, Planned};
pub use recipe::{AdapterInit, Recipe, Step};
pub use sources::Sources;

#[cfg(test)]
pub(crate) mod testutil {
    use ff_core::{DType, Model, Result, TensorInfo, TensorSource};

    /// In-memory source whose tensors are all zero bytes.
    pub struct Fake(pub Model);

    impl TensorSource for Fake {
        fn model(&self) -> &Model {
            &self.0
        }

        fn read(&self, name: &str) -> Result<Vec<u8>> {
            Ok(vec![0; self.0.tensor(name)?.byte_len()])
        }

        fn read_range(&self, name: &str, start: u64, len: u64) -> Result<Vec<u8>> {
            ff_core::ports::slice_range(&self.read(name)?, name, start, len)
        }
    }

    pub fn model(tensors: &[(&str, &[usize])]) -> Model {
        let mut m = Model::default();
        for (name, shape) in tensors {
            m.insert(TensorInfo { name: (*name).into(), dtype: DType::F32, shape: shape.to_vec() });
        }
        m
    }
}
