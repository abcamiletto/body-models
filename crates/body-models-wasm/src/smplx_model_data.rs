//! SMPL-X model data container for Rust CPU forward pass.

/// Parsed SMPL-X model data ready for forward pass computation.
pub struct SmplxModelData {
    pub num_vertices: usize,
    pub num_joints: usize,
    pub num_shape_params: usize,
    pub num_expr_params: usize,
    pub num_pose_params: usize,
    pub rest_pose_y_offset: f32,

    /// Template vertices [V*3], row-major
    pub v_template: Vec<f32>,
    /// Shape blend shapes [V*3*S], indexed as [v*3*S + d*S + s]
    pub shapedirs: Vec<f32>,
    /// Expression blend shapes [V*3*E], indexed as [v*3*E + d*E + e]
    pub exprdirs: Vec<f32>,
    /// Pose blend shapes [P*V*3], indexed as [p*V*3 + v*3 + d]
    pub posedirs: Vec<f32>,

    /// Precomputed joint template positions [J*3]
    pub j_template: Vec<f32>,
    /// Precomputed joint shape directions [J*3*S]
    pub j_shapedirs: Vec<f32>,
    /// Precomputed joint expression directions [J*3*E]
    pub j_exprdirs: Vec<f32>,

    /// Parent joint indices [J], parents[0] = -1
    pub parents: Vec<i32>,
    /// Dense LBS weights [V*J]
    pub lbs_weights: Vec<f32>,
    /// Triangle faces [F*3]
    pub faces: Vec<u32>,

    /// Hand pose means [2*45] (left then right)
    pub hand_mean: Vec<f32>,
}
