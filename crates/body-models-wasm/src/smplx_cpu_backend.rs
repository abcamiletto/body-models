//! CPU forward pass for SMPL-X.

use nalgebra::{Matrix3, Matrix4, Vector3};

use crate::forward_kinematics::{self, KinematicFront};
use crate::rodrigues;
use crate::smplx_model_data::SmplxModelData;

const BODY_POSE_LEN: usize = 21 * 3;
const HAND_POSE_LEN: usize = 30 * 3;
const HEAD_POSE_LEN: usize = 3 * 3;

pub struct SmplxModel {
    pub data: SmplxModelData,
    pub kinematic_fronts: Vec<KinematicFront>,
}

fn assert_opt_batch3(name: &str, value: Option<&[f32]>, batch_size: usize) {
    if let Some(v) = value {
        assert!(
            v.len() == batch_size * 3,
            "{} length {} must be exactly batch_size*3 ({})",
            name,
            v.len(),
            batch_size * 3
        );
    }
}

fn parse_batch_len(total: usize, per: usize, name: &str) -> usize {
    assert!(per > 0, "{}-per-instance must be > 0", name);
    assert!(total > 0 && total % per == 0, "{} length {} must be a positive multiple of {}", name, total, per);
    total / per
}

fn compute_global_transform(
    global_rotation: Option<&[f32]>,
    global_translation: Option<&[f32]>,
    ground_plane: bool,
    rest_pose_y_offset: f32,
) -> (Option<Matrix3<f32>>, Vector3<f32>) {
    let r_global = global_rotation.map(|gr| {
        assert!(gr.len() == 3, "global_rotation length must be 3");
        rodrigues::axis_angle_to_matrix(&Vector3::new(gr[0], gr[1], gr[2]))
    });
    if let Some(gt) = global_translation {
        assert!(gt.len() == 3, "global_translation length must be 3");
    }
    let gt = global_translation.unwrap_or(&[0.0, 0.0, 0.0]);
    let y_off = if ground_plane { rest_pose_y_offset } else { 0.0 };
    (r_global, Vector3::new(gt[0], gt[1] + y_off, gt[2]))
}

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

fn extract_rt(m: &Matrix4<f32>) -> (Matrix3<f32>, Vector3<f32>) {
    (
        m.fixed_view::<3, 3>(0, 0).into_owned(),
        Vector3::new(m[(0, 3)], m[(1, 3)], m[(2, 3)]),
    )
}

impl SmplxModel {
    pub fn new(data: SmplxModelData) -> Self {
        let fronts = forward_kinematics::compute_kinematic_fronts(&data.parents);
        SmplxModel { data, kinematic_fronts: fronts }
    }

    #[cfg(feature = "npz")]
    pub fn from_npz(bytes: &[u8]) -> Result<Self, crate::smplx_npz_loader::NpzError> {
        let data = crate::smplx_npz_loader::load_npz(bytes)?;
        Ok(Self::new(data))
    }

    pub fn forward_vertices(
        &self,
        shape: &[f32],
        body_pose: &[f32],
        hand_pose: &[f32],
        head_pose: &[f32],
        expression: Option<&[f32]>,
        pelvis_rotation: Option<&[f32]>,
        global_rotation: Option<&[f32]>,
        global_translation: Option<&[f32]>,
        ground_plane: bool,
    ) -> Vec<f32> {
        let d = &self.data;
        let b_body = parse_batch_len(body_pose.len(), BODY_POSE_LEN, "body_pose");
        let b_hand = parse_batch_len(hand_pose.len(), HAND_POSE_LEN, "hand_pose");
        let b_head = parse_batch_len(head_pose.len(), HEAD_POSE_LEN, "head_pose");
        assert!(b_body == b_hand && b_body == b_head, "body/hand/head batch sizes must match");
        let b = b_body;
        assert!(shape.len() % b == 0, "shape length must be divisible by batch size");
        assert_opt_batch3("pelvis_rotation", pelvis_rotation, b);
        assert_opt_batch3("global_rotation", global_rotation, b);
        assert_opt_batch3("global_translation", global_translation, b);

        let s_per = shape.len() / b;
        let s_used = s_per.min(d.num_shape_params);
        let (e_per, e_used) = if let Some(expr) = expression {
            assert!(expr.len() % b == 0, "expression length must be divisible by batch size");
            let per = expr.len() / b;
            (per, per.min(d.num_expr_params))
        } else {
            (10usize, 10usize.min(d.num_expr_params))
        };

        let j = d.num_joints;
        let v = d.num_vertices;
        let p = d.num_pose_params;
        assert!(j == 55, "SMPL-X forward currently expects 55 joints, got {}", j);
        assert!(p == (j - 1) * 9, "num_pose_params must equal (J-1)*9");
        assert!(d.hand_mean.len() == 90, "hand_mean must contain 90 values");

        let mut out = Vec::with_capacity(b * v * 3);
        for bi in 0..b {
            let shape_i = &shape[bi * s_per..(bi + 1) * s_per];
            let body_i = &body_pose[bi * BODY_POSE_LEN..(bi + 1) * BODY_POSE_LEN];
            let hand_i = &hand_pose[bi * HAND_POSE_LEN..(bi + 1) * HAND_POSE_LEN];
            let head_i = &head_pose[bi * HEAD_POSE_LEN..(bi + 1) * HEAD_POSE_LEN];
            let expr_i = expression.map(|e| &e[bi * e_per..(bi + 1) * e_per]);
            let pelvis_i = pelvis_rotation.map(|pr| &pr[bi * 3..(bi + 1) * 3]);
            let gro_i = global_rotation.map(|gr| &gr[bi * 3..(bi + 1) * 3]);
            let gtr_i = global_translation.map(|gt| &gt[bi * 3..(bi + 1) * 3]);

            let mut full_pose = vec![0.0f32; j * 3];
            if let Some(pr) = pelvis_i {
                full_pose[..3].copy_from_slice(pr);
            }
            full_pose[3..(3 + BODY_POSE_LEN)].copy_from_slice(body_i);
            full_pose[(3 + BODY_POSE_LEN)..(3 + BODY_POSE_LEN + HEAD_POSE_LEN)].copy_from_slice(head_i);
            for i in 0..45 {
                full_pose[3 + BODY_POSE_LEN + HEAD_POSE_LEN + i] = hand_i[i] + d.hand_mean[i];
                full_pose[3 + BODY_POSE_LEN + HEAD_POSE_LEN + 45 + i] = hand_i[45 + i] + d.hand_mean[45 + i];
            }

            let pose_mats = rodrigues::batch_axis_angle_to_matrix(&full_pose, j);

            let mut j_t = vec![0.0f32; j * 3];
            for ji in 0..j {
                for di in 0..3 {
                    let mut val = d.j_template[ji * 3 + di];
                    for si in 0..s_used {
                        val += shape_i[si] * d.j_shapedirs[ji * 3 * d.num_shape_params + di * d.num_shape_params + si];
                    }
                    if let Some(ex) = expr_i {
                        for ei in 0..e_used {
                            val += ex[ei] * d.j_exprdirs[ji * 3 * d.num_expr_params + di * d.num_expr_params + ei];
                        }
                    }
                    j_t[ji * 3 + di] = val;
                }
            }

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
            let t_world = forward_kinematics::forward_kinematics(&pose_mats, &translations, &self.kinematic_fronts);

            let mut pose_delta = vec![0.0f32; p];
            for ji in 1..j {
                let r = &pose_mats[ji];
                for row in 0..3 {
                    for col in 0..3 {
                        let eye = if row == col { 1.0 } else { 0.0 };
                        pose_delta[(ji - 1) * 9 + row * 3 + col] = r[(row, col)] - eye;
                    }
                }
            }

            let mut v_t = vec![0.0f32; v * 3];
            for vi in 0..v {
                for di in 0..3 {
                    let mut val = d.v_template[vi * 3 + di];
                    for si in 0..s_used {
                        val += shape_i[si] * d.shapedirs[vi * 3 * d.num_shape_params + di * d.num_shape_params + si];
                    }
                    if let Some(ex) = expr_i {
                        for ei in 0..e_used {
                            val += ex[ei] * d.exprdirs[vi * 3 * d.num_expr_params + di * d.num_expr_params + ei];
                        }
                    }
                    v_t[vi * 3 + di] = val;
                }
            }

            for vi in 0..v {
                for di in 0..3 {
                    let vid = vi * 3 + di;
                    let mut acc = 0.0f32;
                    for pi in 0..p {
                        acc += pose_delta[pi] * d.posedirs[pi * v * 3 + vid];
                    }
                    v_t[vid] += acc;
                }
            }

            let mut r_world = Vec::with_capacity(j);
            let mut t_offset = Vec::with_capacity(j);
            for ji in 0..j {
                let (r, t) = extract_rt(&t_world[ji]);
                let jp = Vector3::new(j_t[ji * 3], j_t[ji * 3 + 1], j_t[ji * 3 + 2]);
                r_world.push(r);
                t_offset.push(t - r * jp);
            }
            let (r_global, t_global) =
                compute_global_transform(gro_i, gtr_i, ground_plane, d.rest_pose_y_offset);
            apply_global_transform(&mut r_world, &mut t_offset, r_global.as_ref(), &t_global);

            for vi in 0..v {
                let mut wr = Matrix3::zeros();
                let mut wt = Vector3::zeros();
                for ji in 0..j {
                    let w = d.lbs_weights[vi * j + ji];
                    wr += w * r_world[ji];
                    wt += w * t_offset[ji];
                }
                let vs = Vector3::new(v_t[vi * 3], v_t[vi * 3 + 1], v_t[vi * 3 + 2]);
                let vp = wr * vs + wt;
                out.push(vp[0]);
                out.push(vp[1]);
                out.push(vp[2]);
            }
        }
        out
    }

    pub fn forward_skeleton(
        &self,
        shape: &[f32],
        body_pose: &[f32],
        hand_pose: &[f32],
        head_pose: &[f32],
        expression: Option<&[f32]>,
        pelvis_rotation: Option<&[f32]>,
        global_rotation: Option<&[f32]>,
        global_translation: Option<&[f32]>,
        ground_plane: bool,
    ) -> Vec<f32> {
        let d = &self.data;
        let b_body = parse_batch_len(body_pose.len(), BODY_POSE_LEN, "body_pose");
        let b_hand = parse_batch_len(hand_pose.len(), HAND_POSE_LEN, "hand_pose");
        let b_head = parse_batch_len(head_pose.len(), HEAD_POSE_LEN, "head_pose");
        assert!(b_body == b_hand && b_body == b_head, "body/hand/head batch sizes must match");
        let b = b_body;
        assert!(shape.len() % b == 0, "shape length must be divisible by batch size");
        assert_opt_batch3("pelvis_rotation", pelvis_rotation, b);
        assert_opt_batch3("global_rotation", global_rotation, b);
        assert_opt_batch3("global_translation", global_translation, b);

        let s_per = shape.len() / b;
        let s_used = s_per.min(d.num_shape_params);
        let (e_per, e_used) = if let Some(expr) = expression {
            assert!(expr.len() % b == 0, "expression length must be divisible by batch size");
            let per = expr.len() / b;
            (per, per.min(d.num_expr_params))
        } else {
            (10usize, 10usize.min(d.num_expr_params))
        };
        let j = d.num_joints;
        assert!(j == 55, "SMPL-X forward currently expects 55 joints, got {}", j);
        assert!(d.hand_mean.len() == 90, "hand_mean must contain 90 values");

        let mut out = Vec::with_capacity(b * j * 16);
        for bi in 0..b {
            let shape_i = &shape[bi * s_per..(bi + 1) * s_per];
            let body_i = &body_pose[bi * BODY_POSE_LEN..(bi + 1) * BODY_POSE_LEN];
            let hand_i = &hand_pose[bi * HAND_POSE_LEN..(bi + 1) * HAND_POSE_LEN];
            let head_i = &head_pose[bi * HEAD_POSE_LEN..(bi + 1) * HEAD_POSE_LEN];
            let expr_i = expression.map(|e| &e[bi * e_per..(bi + 1) * e_per]);
            let pelvis_i = pelvis_rotation.map(|pr| &pr[bi * 3..(bi + 1) * 3]);
            let gro_i = global_rotation.map(|gr| &gr[bi * 3..(bi + 1) * 3]);
            let gtr_i = global_translation.map(|gt| &gt[bi * 3..(bi + 1) * 3]);

            let mut full_pose = vec![0.0f32; j * 3];
            if let Some(pr) = pelvis_i {
                full_pose[..3].copy_from_slice(pr);
            }
            full_pose[3..(3 + BODY_POSE_LEN)].copy_from_slice(body_i);
            full_pose[(3 + BODY_POSE_LEN)..(3 + BODY_POSE_LEN + HEAD_POSE_LEN)].copy_from_slice(head_i);
            for i in 0..45 {
                full_pose[3 + BODY_POSE_LEN + HEAD_POSE_LEN + i] = hand_i[i] + d.hand_mean[i];
                full_pose[3 + BODY_POSE_LEN + HEAD_POSE_LEN + 45 + i] = hand_i[45 + i] + d.hand_mean[45 + i];
            }
            let pose_mats = rodrigues::batch_axis_angle_to_matrix(&full_pose, j);

            let mut j_t = vec![0.0f32; j * 3];
            for ji in 0..j {
                for di in 0..3 {
                    let mut val = d.j_template[ji * 3 + di];
                    for si in 0..s_used {
                        val += shape_i[si] * d.j_shapedirs[ji * 3 * d.num_shape_params + di * d.num_shape_params + si];
                    }
                    if let Some(ex) = expr_i {
                        for ei in 0..e_used {
                            val += ex[ei] * d.j_exprdirs[ji * 3 * d.num_expr_params + di * d.num_expr_params + ei];
                        }
                    }
                    j_t[ji * 3 + di] = val;
                }
            }

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
            let t_world = forward_kinematics::forward_kinematics(&pose_mats, &translations, &self.kinematic_fronts);

            let mut r_world = Vec::with_capacity(j);
            let mut t_pos = Vec::with_capacity(j);
            for ji in 0..j {
                let (r, t) = extract_rt(&t_world[ji]);
                r_world.push(r);
                t_pos.push(t);
            }
            let (r_global, t_global) =
                compute_global_transform(gro_i, gtr_i, ground_plane, d.rest_pose_y_offset);
            apply_global_transform(&mut r_world, &mut t_pos, r_global.as_ref(), &t_global);

            for ji in 0..j {
                let r = &r_world[ji];
                let t = &t_pos[ji];
                out.extend_from_slice(&[
                    r[(0, 0)],
                    r[(0, 1)],
                    r[(0, 2)],
                    t[0],
                    r[(1, 0)],
                    r[(1, 1)],
                    r[(1, 2)],
                    t[1],
                    r[(2, 0)],
                    r[(2, 1)],
                    r[(2, 2)],
                    t[2],
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]);
            }
        }
        out
    }
}
