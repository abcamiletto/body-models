//! SMPL body model in Rust, targeting WebAssembly via wgpu.
//!
//! CPU forward pass uses nalgebra for joint-level operations.
//! GPU path (behind `gpu` feature) dispatches vertex-parallel WGSL compute shaders.

pub mod model_data;
pub mod rodrigues;
pub mod forward_kinematics;
pub mod cpu_backend;
pub mod smplx_model_data;
pub mod smplx_cpu_backend;
pub mod sparse_lbs;

#[cfg(feature = "npz")]
pub mod npz_loader;
#[cfg(feature = "npz")]
pub mod smplx_npz_loader;

#[cfg(feature = "gpu")]
pub mod gpu_backend;
#[cfg(feature = "gpu")]
pub mod gpu_forward;

#[cfg(feature = "js")]
pub mod js_api;

pub use cpu_backend::SmplModel;
pub use smplx_cpu_backend::SmplxModel;
