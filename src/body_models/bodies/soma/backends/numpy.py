"""NumPy SOMA backend."""

from . import scipy

__all__ = [
    "apply_pose_correctives",
    "fit_rigid_transform",
    "forward_skeleton",
    "forward_vertices",
    "linear_blend_skinning",
    "SomaIdentity",
    "prepare_data",
    "prepare_pose",
    "prepare_identity_from_rest_shape",
]

apply_pose_correctives = scipy.apply_pose_correctives
fit_rigid_transform = scipy.fit_rigid_transform
forward_skeleton = scipy.forward_skeleton
forward_vertices = scipy.forward_vertices
linear_blend_skinning = scipy.linear_blend_skinning
SomaIdentity = scipy.SomaIdentity
SomaPreparedPose = scipy.SomaPreparedPose
prepare_data = scipy.prepare_data
prepare_pose = scipy.prepare_pose
prepare_identity_from_rest_shape = scipy.prepare_identity_from_rest_shape
