//! Axis-angle to rotation matrix conversion (Rodrigues formula).
//!
//! Matches the nanomanifold pipeline: axis_angle → quaternion → matrix.
//! For small angles, uses Taylor expansion to avoid numerical issues.

use nalgebra::{Matrix3, Vector3};

/// Small angle threshold: sqrt(f32::EPSILON) ≈ 3.45e-4
const SMALL_ANGLE_THRESH: f32 = 3.4526698e-4; // sqrt(1.1920929e-7)

/// Convert axis-angle [3] to 3x3 rotation matrix.
///
/// Uses the quaternion path matching nanomanifold:
///   axis_angle → quaternion (w, x, y, z) → rotation matrix
///
/// For small angles (||axis_angle|| < sqrt(eps)):
///   half_angle ≈ 0, so cos(half) ≈ 1, sin(half)/angle ≈ 0.5
///   quaternion ≈ (1, axis_angle/2)
pub fn axis_angle_to_matrix(axis_angle: &Vector3<f32>) -> Matrix3<f32> {
    let angle = axis_angle.norm();

    if angle < SMALL_ANGLE_THRESH {
        // Small-angle: quaternion ≈ (1, aa/2), matching nanomanifold's Taylor path
        let hx = axis_angle[0] * 0.5;
        let hy = axis_angle[1] * 0.5;
        let hz = axis_angle[2] * 0.5;
        // w = 1 (approximately), but we normalize for accuracy
        quat_to_matrix(1.0, hx, hy, hz)
    } else {
        let half_angle = angle * 0.5;
        let cos_half = half_angle.cos();
        let sin_half = half_angle.sin();
        let axis = axis_angle / angle;

        let w = cos_half;
        let x = sin_half * axis[0];
        let y = sin_half * axis[1];
        let z = sin_half * axis[2];
        quat_to_matrix(w, x, y, z)
    }
}

/// Convert quaternion (w, x, y, z) to rotation matrix.
/// Matches nanomanifold's `to_matrix` exactly.
#[inline]
fn quat_to_matrix(w: f32, x: f32, y: f32, z: f32) -> Matrix3<f32> {
    Matrix3::new(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),       2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y),
    )
}

/// Batch convert axis-angle vectors to rotation matrices.
/// Input: slice of 3-element axis-angle vectors (flat: [j*3+d])
/// Output: Vec of 3x3 rotation matrices
pub fn batch_axis_angle_to_matrix(axis_angles: &[f32], num_joints: usize) -> Vec<Matrix3<f32>> {
    assert_eq!(axis_angles.len(), num_joints * 3);
    let mut matrices = Vec::with_capacity(num_joints);
    for j in 0..num_joints {
        let aa = Vector3::new(
            axis_angles[j * 3],
            axis_angles[j * 3 + 1],
            axis_angles[j * 3 + 2],
        );
        matrices.push(axis_angle_to_matrix(&aa));
    }
    matrices
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity_rotation() {
        let aa = Vector3::new(0.0, 0.0, 0.0);
        let r = axis_angle_to_matrix(&aa);
        assert_relative_eq!(r, Matrix3::identity(), epsilon = 1e-6);
    }

    #[test]
    fn test_90deg_z() {
        let aa = Vector3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2);
        let r = axis_angle_to_matrix(&aa);
        // Should map x→y, y→-x
        let expected = Matrix3::new(
            0.0, -1.0, 0.0,
            1.0,  0.0, 0.0,
            0.0,  0.0, 1.0,
        );
        assert_relative_eq!(r, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_180deg_x() {
        let aa = Vector3::new(std::f32::consts::PI, 0.0, 0.0);
        let r = axis_angle_to_matrix(&aa);
        let expected = Matrix3::new(
            1.0,  0.0,  0.0,
            0.0, -1.0,  0.0,
            0.0,  0.0, -1.0,
        );
        assert_relative_eq!(r, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_small_angle() {
        let aa = Vector3::new(1e-5, 2e-5, 3e-5);
        let r = axis_angle_to_matrix(&aa);
        // Should be very close to identity
        assert_relative_eq!(r, Matrix3::identity(), epsilon = 1e-4);
    }
}
