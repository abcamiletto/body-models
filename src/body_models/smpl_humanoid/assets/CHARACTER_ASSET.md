# Professional rigid SMPL character

`smpl_robot_professional.blend` and `smpl_robot_professional.glb` are an
original classic mannequin design authored for this project using Blender 5.0
and Blender MCP.

The delivered character contains no skin weights, armature, animation,
armature modifiers, shape keys, or blend shapes. It has a sculpted neutral
mannequin head, five modeled digits per hand, bilateral mesh symmetry, and
separate rigid geometry for both the SMPL ankle and toe links. Every exported
mesh name begins with its owning SMPL joint index.

The authoring geometry is vertex-level mirrored across the sagittal plane.
Runtime generation averages and reflects the neutral SMPL-X reference offsets,
then applies SMPL shape as relative bone-length changes. This keeps the rigid
pieces symmetric while matching SMPL-X forward kinematics.

The Blender and GLB files retain the full-resolution authoring geometry. The
distributed runtime levels use curvature-weighted, topology-preserving quadric
simplification. Each bilateral part is optimized once and reflected exactly,
including centerline meshes, so every LOD is sagittally symmetric.

The character uses exactly two closely related matte pastel materials. Edit
the armor and joint materials in the bundled Blender file to restyle every
shell and joint surface.
