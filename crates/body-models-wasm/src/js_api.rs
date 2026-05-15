//! wasm-bindgen exports for browser usage.

use wasm_bindgen::prelude::*;

use crate::cpu_backend::SmplModel;

#[wasm_bindgen]
pub struct WasmSmplModel {
    model: SmplModel,
}

#[wasm_bindgen]
impl WasmSmplModel {
    /// Load model from .npz or .bin bytes. Format is auto-detected.
    #[wasm_bindgen(constructor)]
    pub fn new(data: &[u8]) -> Result<WasmSmplModel, JsError> {
        #[cfg(feature = "js")]
        console_error_panic_hook::set_once();

        let model = SmplModel::from_bytes(data)
            .map_err(|e| JsError::new(&format!("Failed to parse model: {}", e)))?;
        Ok(WasmSmplModel { model })
    }

    /// Number of mesh vertices.
    #[wasm_bindgen(getter, js_name = "numVertices")]
    pub fn num_vertices(&self) -> u32 {
        self.model.data.num_vertices as u32
    }

    /// Number of joints.
    #[wasm_bindgen(getter, js_name = "numJoints")]
    pub fn num_joints(&self) -> u32 {
        self.model.data.num_joints as u32
    }

    /// Number of shape parameters.
    #[wasm_bindgen(getter, js_name = "numShapeParams")]
    pub fn num_shape_params(&self) -> u32 {
        self.model.data.num_shape_params as u32
    }

    /// Number of faces.
    #[wasm_bindgen(getter, js_name = "numFaces")]
    pub fn num_faces(&self) -> u32 {
        self.model.data.num_faces as u32
    }

    /// Get face indices as Uint32Array [F*3].
    #[wasm_bindgen(js_name = "getFaces")]
    pub fn get_faces(&self) -> Vec<u32> {
        self.model.data.faces.clone()
    }

    /// Get rest-pose template vertices as Float32Array [V*3].
    #[wasm_bindgen(js_name = "getTemplateVertices")]
    pub fn get_template_vertices(&self) -> Vec<f32> {
        self.model.data.v_template.clone()
    }

    /// Compute forward vertices (supports batching).
    ///
    /// Batch size B is inferred from body_pose.length / 69.
    /// When B=1, behavior is identical to the single-instance API.
    ///
    /// Args:
    ///   shape: Float32Array[B*S] - S shape parameters per instance (e.g. S=10)
    ///   body_pose: Float32Array[B*69] - 23*3 body joint axis-angle rotations per instance
    ///   pelvis_rotation: Float32Array[B*3] or undefined - pelvis rotation per instance
    ///   global_translation: Float32Array[B*3] or undefined - global translation per instance
    ///   ground_plane: boolean - whether to apply ground plane offset
    ///
    /// Returns: Float32Array[B*V*3] - vertex positions
    #[wasm_bindgen(js_name = "forwardVertices")]
    pub fn forward_vertices(
        &self,
        shape: &[f32],
        body_pose: &[f32],
        pelvis_rotation: Option<Vec<f32>>,
        global_translation: Option<Vec<f32>>,
        ground_plane: bool,
    ) -> Vec<f32> {
        self.model.forward_vertices(
            shape,
            body_pose,
            pelvis_rotation.as_deref(),
            None,
            global_translation.as_deref(),
            ground_plane,
        )
    }

    /// Compute forward skeleton transforms (supports batching).
    ///
    /// Batch size B is inferred from body_pose.length / 69.
    ///
    /// Returns: Float32Array[B*J*16] - 4x4 row-major transforms per joint per instance
    #[wasm_bindgen(js_name = "forwardSkeleton")]
    pub fn forward_skeleton(
        &self,
        shape: &[f32],
        body_pose: &[f32],
        pelvis_rotation: Option<Vec<f32>>,
        global_translation: Option<Vec<f32>>,
        ground_plane: bool,
    ) -> Vec<f32> {
        self.model.forward_skeleton(
            shape,
            body_pose,
            pelvis_rotation.as_deref(),
            None,
            global_translation.as_deref(),
            ground_plane,
        )
    }
}
