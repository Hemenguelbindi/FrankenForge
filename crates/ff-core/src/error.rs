use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{}: {source}", .path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{}: invalid model file: {reason}", .path.display())]
    Format { path: PathBuf, reason: String },
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("incompatible: {0}")]
    Incompatible(String),
    #[error("recipe: {0}")]
    Recipe(String),
    #[error("verification failed: {0}")]
    Verify(String),
    /// The structure changes but config.json cannot be updated completely: refuse rather than
    /// produce a model whose weights and config disagree.
    #[error("unsupported config: {0}")]
    UnsupportedConfig(String),
    /// The build uses an experimental feature (adapter) without explicit opt-in.
    #[error("experimental: {0}")]
    Experimental(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Adapter for `map_err` that attaches a path to an I/O error.
    pub fn io(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> Error {
        let path = path.into();
        move |source| Error::Io { path, source }
    }
}
