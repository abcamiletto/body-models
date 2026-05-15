//! Forward kinematics using precomputed kinematic fronts.
//!
//! Matches the Python `compute_kinematic_fronts` + `_batched_forward_kinematics`.

use nalgebra::{Matrix3, Matrix4, Vector3};

/// A kinematic front: a set of joints at the same depth, with their parent indices.
/// For the root (depth 0), parent_indices[i] == -1.
#[derive(Debug, Clone)]
pub struct KinematicFront {
    pub joint_indices: Vec<usize>,
    pub parent_indices: Vec<i32>,
}

/// Compute kinematic fronts from parent array.
/// Matches Python `compute_kinematic_fronts`.
pub fn compute_kinematic_fronts(parents: &[i32]) -> Vec<KinematicFront> {
    let n_joints = parents.len();
    let mut depths = vec![-1i32; n_joints];
    depths[0] = 0;

    for i in 1..n_joints {
        let mut d = 0;
        let mut j = i;
        while j != 0 {
            j = parents[j] as usize;
            d += 1;
        }
        depths[i] = d;
    }

    let max_depth = *depths.iter().max().unwrap();
    let mut fronts = Vec::new();

    for d in 0..=max_depth {
        let joints: Vec<usize> = (0..n_joints).filter(|&i| depths[i] == d).collect();
        let parent_indices: Vec<i32> = if d == 0 {
            vec![-1; joints.len()]
        } else {
            joints.iter().map(|&j| parents[j]).collect()
        };
        fronts.push(KinematicFront { joint_indices: joints, parent_indices });
    }

    fronts
}

/// Build a 4x4 homogeneous transform from rotation and translation.
#[inline]
pub fn build_transform(r: &Matrix3<f32>, t: &Vector3<f32>) -> Matrix4<f32> {
    Matrix4::new(
        r[(0, 0)], r[(0, 1)], r[(0, 2)], t[0],
        r[(1, 0)], r[(1, 1)], r[(1, 2)], t[1],
        r[(2, 0)], r[(2, 1)], r[(2, 2)], t[2],
        0.0,       0.0,       0.0,       1.0,
    )
}

/// Batched forward kinematics using precomputed kinematic fronts.
///
/// Inputs:
///   - `rotations`: [J] local rotation matrices (from Rodrigues)
///   - `translations`: [J] local translation vectors
///   - `fronts`: precomputed kinematic fronts
///
/// Output: [J] world-space 4x4 transforms
pub fn forward_kinematics(
    rotations: &[Matrix3<f32>],
    translations: &[Vector3<f32>],
    fronts: &[KinematicFront],
) -> Vec<Matrix4<f32>> {
    let n_joints = rotations.len();
    let mut t_world = vec![Matrix4::identity(); n_joints];

    // Build local transforms
    let t_local: Vec<Matrix4<f32>> = rotations.iter().zip(translations.iter())
        .map(|(r, t)| build_transform(r, t))
        .collect();

    for front in fronts {
        for (idx, &joint) in front.joint_indices.iter().enumerate() {
            let parent_idx = front.parent_indices[idx];
            if parent_idx < 0 {
                // Root joint
                t_world[joint] = t_local[joint];
            } else {
                // T_world[joint] = T_world[parent] * T_local[joint]
                t_world[joint] = t_world[parent_idx as usize] * t_local[joint];
            }
        }
    }

    t_world
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fronts_smpl() {
        // SMPL parent array (24 joints)
        let parents = [
            -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
        ];
        let fronts = compute_kinematic_fronts(&parents);

        // Root should be alone at depth 0
        assert_eq!(fronts[0].joint_indices, vec![0]);
        assert_eq!(fronts[0].parent_indices, vec![-1]);

        // All joints should appear exactly once
        let mut all_joints: Vec<usize> = fronts.iter()
            .flat_map(|f| f.joint_indices.clone())
            .collect();
        all_joints.sort();
        assert_eq!(all_joints, (0..24).collect::<Vec<_>>());
    }

    #[test]
    fn test_identity_fk() {
        let parents = [-1, 0, 0, 1];
        let fronts = compute_kinematic_fronts(&parents);

        let rotations = vec![Matrix3::identity(); 4];
        let translations = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.5, 0.0, 0.0),
        ];

        let transforms = forward_kinematics(&rotations, &translations, &fronts);

        // Joint 3: parent is 1, so position = t[1] + t[3] = (1.5, 0, 0)
        let pos3 = transforms[3].column(3).xyz();
        assert!((pos3 - Vector3::new(1.5, 0.0, 0.0)).norm() < 1e-6);
    }
}
