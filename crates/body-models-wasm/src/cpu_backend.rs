//! CPU forward pass for SMPL.
//!
//! Implements forward_vertices and forward_skeleton matching
//! the Python `core.py` computation exactly.

use nalgebra::{Matrix3, Matrix4, Vector3};

use crate::forward_kinematics::{self, KinematicFront};
use crate::model_data::SmplModelData;
use crate::rodrigues;

/// Intermediate result from joint-level computation.
/// Shared between forward_vertices, forward_skeleton, and GPU path.
pub struct JointData {
    pub pose_matrices: Vec<Matrix3<f32>>,
    pub j_t: Vec<f32>,              // [J*3] shaped joint positions
    pub t_world: Vec<Matrix4<f32>>, // [J] FK world-space 4x4 transforms
}

/// Compute pose delta: (pose_matrices[1:] - I).flatten() → [(J-1)*9]
pub fn compute_pose_delta(pose_matrices: &[Matrix3<f32>], num_joints: usize, num_pose_params: usize) -> Vec<f32> {
    let mut pose_delta = vec![0.0f32; num_pose_params];
    for ji in 1..num_joints {
        let r = &pose_matrices[ji];
        for row in 0..3 {
            for col in 0..3 {
                let eye_val = if row == col { 1.0 } else { 0.0 };
                pose_delta[(ji - 1) * 9 + row * 3 + col] = r[(row, col)] - eye_val;
            }
        }
    }
    pose_delta
}

/// Extract rotation matrix and translation vector from a 4x4 transform.
fn extract_rt(m: &Matrix4<f32>) -> (Matrix3<f32>, Vector3<f32>) {
    (
        m.fixed_view::<3, 3>(0, 0).into_owned(),
        Vector3::new(m[(0, 3)], m[(1, 3)], m[(2, 3)]),
    )
}

/// Compute global rotation matrix and translation vector (with y_offset fused).
fn compute_global_transform(
    global_rotation: Option<&[f32]>,
    global_translation: Option<&[f32]>,
    ground_plane: bool,
    rest_pose_y_offset: f32,
) -> (Option<Matrix3<f32>>, Vector3<f32>) {
    let r_global = global_rotation.map(|gr|
        rodrigues::axis_angle_to_matrix(&Vector3::new(gr[0], gr[1], gr[2]))
    );
    let gt = global_translation.unwrap_or(&[0.0, 0.0, 0.0]);
    let y_off = if ground_plane { rest_pose_y_offset } else { 0.0 };
    (r_global, Vector3::new(gt[0], gt[1] + y_off, gt[2]))
}

/// Apply global rotation + translation to per-joint transforms in place.
fn apply_global_transform(
    r_world: &mut [Matrix3<f32>],
    t_vec: &mut [Vector3<f32>],
    r_global: Option<&Matrix3<f32>>,
    t_global: &Vector3<f32>,
) {
    for ji in 0..r_world.len() {
        if let Some(rg) = r_global {
            r_world[ji] = rg * r_world[ji];
            t_vec[ji] = rg * t_vec[ji];
        }
        t_vec[ji] += t_global;
    }
}

/// Compute LBS-ready per-joint transforms from FK results.
///
/// Returns `(r_world[J], t_offset[J])` where `t_offset = t_fk - R * j_pos`,
/// with global rotation/translation/y_offset fused in.
/// Used by both CPU `forward_vertices` and GPU `gpu_forward`.
pub fn compute_lbs_transforms(
    jd: &JointData,
    data: &SmplModelData,
    global_rotation: Option<&[f32]>,
    global_translation: Option<&[f32]>,
    ground_plane: bool,
) -> (Vec<Matrix3<f32>>, Vec<Vector3<f32>>) {
    let j = data.num_joints;
    let mut r_world = Vec::with_capacity(j);
    let mut t_offset = Vec::with_capacity(j);

    for ji in 0..j {
        let (r, t) = extract_rt(&jd.t_world[ji]);
        let j_pos = Vector3::new(jd.j_t[ji * 3], jd.j_t[ji * 3 + 1], jd.j_t[ji * 3 + 2]);
        t_offset.push(t - r * j_pos);
        r_world.push(r);
    }

    let (r_global, t_global) = compute_global_transform(
        global_rotation, global_translation, ground_plane, data.rest_pose_y_offset,
    );
    apply_global_transform(&mut r_world, &mut t_offset, r_global.as_ref(), &t_global);

    (r_world, t_offset)
}

/// Precomputed model ready for forward pass.
pub struct SmplModel {
    pub data: SmplModelData,
    pub kinematic_fronts: Vec<KinematicFront>,
}

impl SmplModel {
    pub fn new(data: SmplModelData) -> Self {
        let fronts = forward_kinematics::compute_kinematic_fronts(&data.parents);
        SmplModel { data, kinematic_fronts: fronts }
    }

    pub fn from_bin(bytes: &[u8]) -> Result<Self, crate::model_data::ParseError> {
        let data = SmplModelData::from_bytes(bytes)?;
        Ok(Self::new(data))
    }

    #[cfg(feature = "npz")]
    pub fn from_npz(bytes: &[u8]) -> Result<Self, crate::npz_loader::NpzError> {
        let data = crate::npz_loader::load_npz(bytes)?;
        Ok(Self::new(data))
    }

    /// Auto-detect format (.bin or .npz) from magic bytes and load.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 4 {
            return Err("Data too short".into());
        }
        let magic = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        if magic == 0x534D504C {
            return Self::from_bin(bytes).map_err(|e| e.to_string());
        }
        #[cfg(feature = "npz")]
        if &bytes[..4] == b"PK\x03\x04" {
            return Self::from_npz(bytes).map_err(|e| e.to_string());
        }
        Err("Unknown format: expected .bin (magic 0x534D504C) or .npz (ZIP)".into())
    }

    /// Number of body pose elements per instance: `(num_joints - 1) * 3`.
    pub fn body_pose_len(&self) -> usize {
        (self.data.num_joints - 1) * 3
    }

    /// Compute joint-level data shared between all forward passes.
    ///
    /// Performs: Rodrigues → joint locations → FK.
    /// Returns pose matrices, shaped joint positions, and world transforms.
    pub fn compute_joints(
        &self,
        shape: &[f32],
        body_pose: &[f32],
        pelvis_rotation: Option<&[f32]>,
    ) -> JointData {
        let d = &self.data;
        let j = d.num_joints;
        let s = d.num_shape_params;
        let s_used = shape.len().min(s);

        // Build full pose: cat([pelvis, body_pose])
        let zero3 = [0.0f32; 3];
        let pelvis = pelvis_rotation.unwrap_or(&zero3);
        let mut full_pose = Vec::with_capacity(j * 3);
        full_pose.extend_from_slice(pelvis);
        full_pose.extend_from_slice(body_pose);

        let pose_matrices = rodrigues::batch_axis_angle_to_matrix(&full_pose, j);

        // Joint locations: j_t = j_template + einsum("p,jdp->jd", shape, j_shapedirs)
        let mut j_t = vec![0.0f32; j * 3];
        for ji in 0..j {
            for di in 0..3 {
                let mut val = d.j_template[ji * 3 + di];
                for si in 0..s_used {
                    val += shape[si] * d.j_shapedirs[ji * 3 * s + di * s + si];
                }
                j_t[ji * 3 + di] = val;
            }
        }

        // FK translations
        let mut translations = Vec::with_capacity(j);
        translations.push(Vector3::new(j_t[0], j_t[1], j_t[2]));
        for ji in 1..j {
            let pi = d.parents[ji] as usize;
            translations.push(Vector3::new(
                j_t[ji * 3] - j_t[pi * 3],
                j_t[ji * 3 + 1] - j_t[pi * 3 + 1],
                j_t[ji * 3 + 2] - j_t[pi * 3 + 2],
            ));
        }

        let t_world = forward_kinematics::forward_kinematics(
            &pose_matrices,
            &translations,
            &self.kinematic_fronts,
        );

        JointData { pose_matrices, j_t, t_world }
    }

    /// Compute mesh vertices for one or more instances.
    ///
    /// Batch size B is inferred from `body_pose.len() / body_pose_len()`.
    /// - `shape`: `[B*S]` (S = shape.len() / B, must divide evenly)
    /// - `body_pose`: `[B*body_pose_len]`
    /// - `pelvis_rotation`: None or `[B*3]`
    /// - `global_rotation`: None or `[B*3]`
    /// - `global_translation`: None or `[B*3]`
    ///
    /// Returns `[B*V*3]` vertex positions.
    pub fn forward_vertices(
        &self,
        shape: &[f32],
        body_pose: &[f32],
        pelvis_rotation: Option<&[f32]>,
        global_rotation: Option<&[f32]>,
        global_translation: Option<&[f32]>,
        ground_plane: bool,
    ) -> Vec<f32> {
        let d = &self.data;
        let v = d.num_vertices;
        let j = d.num_joints;
        let s = d.num_shape_params;
        let p = d.num_pose_params;
        let k = d.nnz_per_vertex;
        let bp_len = self.body_pose_len();

        let batch_size = parse_batch_size(body_pose.len(), bp_len, shape.len());

        let s_per = shape.len() / batch_size;
        let s_used = s_per.min(s);

        let mut v_posed = Vec::with_capacity(batch_size * v * 3);

        for bi in 0..batch_size {
            let (shape_i, body_pose_i, pelvis_i, global_rot_i, global_trans_i) =
                slice_instance(bi, s_per, bp_len, shape, body_pose, pelvis_rotation, global_rotation, global_translation);

            let jd = self.compute_joints(shape_i, body_pose_i, pelvis_i);

            // Shape blend shapes: v_shaped = v_template + shapedirs @ shape
            let mut v_shaped = vec![0.0f32; v * 3];
            for vi in 0..v {
                for di in 0..3 {
                    let mut val = d.v_template[vi * 3 + di];
                    for si in 0..s_used {
                        val += shape_i[si] * d.shapedirs[vi * 3 * s + di * s + si];
                    }
                    v_shaped[vi * 3 + di] = val;
                }
            }

            // Pose blend shapes: v_shaped += posedirs @ pose_delta
            let pose_delta = compute_pose_delta(&jd.pose_matrices, j, p);
            for vi in 0..v {
                for di in 0..3 {
                    let base = vi * 3 * p + di * p;
                    let mut acc = 0.0f32;
                    for pi in 0..p {
                        acc += pose_delta[pi] * d.posedirs[base + pi];
                    }
                    v_shaped[vi * 3 + di] += acc;
                }
            }

            let (r_world, t_offset) = compute_lbs_transforms(
                &jd, d, global_rot_i, global_trans_i, ground_plane,
            );

            // Sparse LBS: v_posed = sum_k(w_k * (R_k @ v_shaped + t_k))
            for vi in 0..v {
                let mut wr = Matrix3::zeros();
                let mut wt = Vector3::zeros();
                for ki in 0..k {
                    let ji = d.sparse_indices[vi * k + ki] as usize;
                    let w = d.sparse_weights[vi * k + ki];
                    wr += w * r_world[ji];
                    wt += w * t_offset[ji];
                }
                let vs = Vector3::new(v_shaped[vi * 3], v_shaped[vi * 3 + 1], v_shaped[vi * 3 + 2]);
                let vp = wr * vs + wt;
                v_posed.push(vp[0]);
                v_posed.push(vp[1]);
                v_posed.push(vp[2]);
            }
        }

        v_posed
    }

    /// Compute skeleton joint transforms for one or more instances.
    ///
    /// Batch size B is inferred from `body_pose.len() / body_pose_len()`.
    /// Returns `[B*J*16]` array of 4x4 row-major transforms.
    pub fn forward_skeleton(
        &self,
        shape: &[f32],
        body_pose: &[f32],
        pelvis_rotation: Option<&[f32]>,
        global_rotation: Option<&[f32]>,
        global_translation: Option<&[f32]>,
        ground_plane: bool,
    ) -> Vec<f32> {
        let d = &self.data;
        let j = d.num_joints;
        let bp_len = self.body_pose_len();

        let batch_size = parse_batch_size(body_pose.len(), bp_len, shape.len());
        let s_per = shape.len() / batch_size;

        let mut result = Vec::with_capacity(batch_size * j * 16);

        for bi in 0..batch_size {
            let (shape_i, body_pose_i, pelvis_i, global_rot_i, global_trans_i) =
                slice_instance(bi, s_per, bp_len, shape, body_pose, pelvis_rotation, global_rotation, global_translation);

            let jd = self.compute_joints(shape_i, body_pose_i, pelvis_i);

            // Extract R and t_pos (world-space joint position) from FK
            let mut r_world = Vec::with_capacity(j);
            let mut t_pos = Vec::with_capacity(j);
            for ji in 0..j {
                let (r, t) = extract_rt(&jd.t_world[ji]);
                r_world.push(r);
                t_pos.push(t);
            }

            let (r_global, t_global) = compute_global_transform(
                global_rot_i, global_trans_i, ground_plane, d.rest_pose_y_offset,
            );
            apply_global_transform(&mut r_world, &mut t_pos, r_global.as_ref(), &t_global);

            // Build output: J * 16 floats (row-major 4x4)
            for ji in 0..j {
                let r = &r_world[ji];
                let t = &t_pos[ji];
                result.extend_from_slice(&[
                    r[(0, 0)], r[(0, 1)], r[(0, 2)], t[0],
                    r[(1, 0)], r[(1, 1)], r[(1, 2)], t[1],
                    r[(2, 0)], r[(2, 1)], r[(2, 2)], t[2],
                    0.0, 0.0, 0.0, 1.0,
                ]);
            }
        }

        result
    }
}

/// Validate batch dimensions and return batch size.
pub fn parse_batch_size(body_pose_len: usize, bp_per: usize, shape_len: usize) -> usize {
    assert!(body_pose_len > 0 && body_pose_len % bp_per == 0,
        "body_pose length {} must be a positive multiple of {}", body_pose_len, bp_per);
    let batch_size = body_pose_len / bp_per;
    assert!(shape_len % batch_size == 0,
        "shape length {} must be divisible by batch size {}", shape_len, batch_size);
    batch_size
}

/// Extract per-instance slices from batched input arrays.
pub fn slice_instance<'a>(
    bi: usize,
    s_per: usize,
    bp_len: usize,
    shape: &'a [f32],
    body_pose: &'a [f32],
    pelvis_rotation: Option<&'a [f32]>,
    global_rotation: Option<&'a [f32]>,
    global_translation: Option<&'a [f32]>,
) -> (&'a [f32], &'a [f32], Option<&'a [f32]>, Option<&'a [f32]>, Option<&'a [f32]>) {
    (
        &shape[bi * s_per..(bi + 1) * s_per],
        &body_pose[bi * bp_len..(bi + 1) * bp_len],
        pelvis_rotation.map(|pr| &pr[bi * 3..(bi + 1) * 3]),
        global_rotation.map(|gr| &gr[bi * 3..(bi + 1) * 3]),
        global_translation.map(|gt| &gt[bi * 3..(bi + 1) * 3]),
    )
}
