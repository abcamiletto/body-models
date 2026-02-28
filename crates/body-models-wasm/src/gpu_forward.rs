//! GPU dispatch orchestration (batched).
//!
//! CPU computes joint-level data for all B instances →
//! packs concatenated buffers → GPU dispatches 3 compute passes →
//! single readback of [B*V*3].

use crate::cpu_backend::{SmplModel, compute_pose_delta, compute_lbs_transforms, parse_batch_size, slice_instance};
use crate::gpu_backend::{GpuBackend, bg_entry, create_init_buf};
use crate::model_data::SmplModelData;

/// GPU-accelerated SMPL forward pass with batch support.
pub struct GpuSmplModel {
    pub model: SmplModel,
    pub gpu: GpuBackend,
}

impl GpuSmplModel {
    pub async fn new(data: SmplModelData) -> Self {
        let gpu = GpuBackend::new(&data).await;
        let model = SmplModel::new(data);
        GpuSmplModel { model, gpu }
    }

    /// Forward vertices using GPU compute shaders.
    ///
    /// Batch size B is inferred from `body_pose.len() / body_pose_len()`.
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
        let d = &self.model.data;
        let j = d.num_joints;
        let s = d.num_shape_params;
        let p = d.num_pose_params;
        let v = d.num_vertices;
        let bp_len = self.model.body_pose_len();

        let batch_size = parse_batch_size(body_pose.len(), bp_len, shape.len());
        let s_per = shape.len() / batch_size;
        let s_used = s_per.min(s);

        // === CPU: Joint-level computation for all B instances ===
        // Pack shape compactly as [B * s_used] — the shader indexes with stride S = s_used
        let mut shape_full = vec![0.0f32; batch_size * s_used];
        let mut all_pose_delta = Vec::with_capacity(batch_size * p);
        let mut all_joint_rt = Vec::with_capacity(batch_size * j * 12);

        for bi in 0..batch_size {
            let (shape_i, body_pose_i, pelvis_i, global_rot_i, global_trans_i) =
                slice_instance(bi, s_per, bp_len, shape, body_pose, pelvis_rotation, global_rotation, global_translation);

            shape_full[bi * s_used..bi * s_used + s_used].copy_from_slice(&shape_i[..s_used]);

            let jd = self.model.compute_joints(shape_i, body_pose_i, pelvis_i);
            all_pose_delta.extend_from_slice(&compute_pose_delta(&jd.pose_matrices, j, p));

            // Pack fused joint R + t_offset into [J*12] for GPU
            let (r_world, t_offset) = compute_lbs_transforms(
                &jd, d, global_rot_i, global_trans_i, ground_plane,
            );
            for ji in 0..j {
                let r = &r_world[ji];
                let t = &t_offset[ji];
                all_joint_rt.extend_from_slice(&[
                    r[(0, 0)], r[(0, 1)], r[(0, 2)],
                    r[(1, 0)], r[(1, 1)], r[(1, 2)],
                    r[(2, 0)], r[(2, 1)], r[(2, 2)],
                    t[0], t[1], t[2],
                ]);
            }
        }

        // === GPU: Create per-call buffers, bind groups, dispatch ===
        let gpu = &self.gpu;
        let b = batch_size;
        use wgpu::BufferUsages as BU;

        let shape_buf = create_init_buf(&gpu.device, "shape", &shape_full, BU::STORAGE);
        let pose_delta_buf = create_init_buf(&gpu.device, "pose_delta", &all_pose_delta, BU::STORAGE);
        let joint_rt_buf = create_init_buf(&gpu.device, "joint_rt", &all_joint_rt, BU::STORAGE);

        let v_shaped_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v_shaped"),
            size: (b * v * 3 * 4) as u64,
            usage: BU::STORAGE,
            mapped_at_creation: false,
        });
        let v_posed_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("v_posed"),
            size: (b * v * 3 * 4) as u64,
            usage: BU::STORAGE | BU::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (b * v * 3 * 4) as u64,
            usage: BU::MAP_READ | BU::COPY_DST,
            mapped_at_creation: false,
        });

        let shape_blend_params: [u32; 4] = [v as u32, s_used as u32, b as u32, 0];
        let shape_blend_params_buf = create_init_buf(&gpu.device, "shape_blend_params", &shape_blend_params, BU::UNIFORM);

        let pose_blend_params: [u32; 4] = [v as u32, p as u32, b as u32, 0];
        let pose_blend_params_buf = create_init_buf(&gpu.device, "pose_blend_params", &pose_blend_params, BU::UNIFORM);

        let lbs_params: [u32; 4] = [v as u32, d.nnz_per_vertex as u32, j as u32, b as u32];
        let lbs_params_buf = create_init_buf(&gpu.device, "lbs_params", &lbs_params, BU::UNIFORM);

        // Bind groups
        let shape_blend_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shape_blend_bg"),
            layout: &gpu.shape_blend_bgl,
            entries: &[
                bg_entry(0, &shape_blend_params_buf),
                bg_entry(1, &gpu.v_template_buf),
                bg_entry(2, &gpu.shapedirs_buf),
                bg_entry(3, &shape_buf),
                bg_entry(4, &v_shaped_buf),
            ],
        });
        let pose_blend_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pose_blend_bg"),
            layout: &gpu.pose_blend_bgl,
            entries: &[
                bg_entry(0, &pose_blend_params_buf),
                bg_entry(1, &gpu.posedirs_buf),
                bg_entry(2, &pose_delta_buf),
                bg_entry(3, &v_shaped_buf),
            ],
        });
        let lbs_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbs_bg"),
            layout: &gpu.lbs_bgl,
            entries: &[
                bg_entry(0, &lbs_params_buf),
                bg_entry(1, &v_shaped_buf),
                bg_entry(2, &gpu.sparse_indices_buf),
                bg_entry(3, &gpu.sparse_weights_buf),
                bg_entry(4, &joint_rt_buf),
                bg_entry(5, &v_posed_buf),
            ],
        });

        // Dispatch compute passes
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("smpl_forward"),
        });

        let wg_bvd = ((b * v * 3 + 63) / 64) as u32;
        let wg_bv = ((b * v + 63) / 64) as u32;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shape_blend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.shape_blend_pipeline);
            pass.set_bind_group(0, &shape_blend_bg, &[]);
            pass.dispatch_workgroups(wg_bvd, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pose_blend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.pose_blend_pipeline);
            pass.set_bind_group(0, &pose_blend_bg, &[]);
            pass.dispatch_workgroups(wg_bvd, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lbs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.lbs_pipeline);
            pass.set_bind_group(0, &lbs_bg, &[]);
            pass.dispatch_workgroups(wg_bv, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &v_posed_buf, 0,
            &staging_buf, 0,
            (b * v * 3 * 4) as u64,
        );

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Readback
        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buf.unmap();

        result
    }
}
