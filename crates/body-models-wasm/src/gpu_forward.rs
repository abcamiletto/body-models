//! GPU dispatch orchestration (batched, pooled buffers).
//!
//! CPU computes joint-level data for all B instances →
//! write_buffer into pooled GPU buffers → 2 compute passes →
//! single readback of [B*V*3].

use crate::cpu_backend::{SmplModel, compute_pose_delta, compute_lbs_transforms, parse_batch_size, slice_instance};
use crate::gpu_backend::GpuBackend;
use crate::model_data::SmplModelData;

/// GPU-accelerated SMPL forward pass with batch support.
pub struct GpuSmplModel {
    pub model: SmplModel,
    pub gpu: GpuBackend,
}

impl GpuSmplModel {
    pub async fn try_new(data: SmplModelData, max_batch_size: usize) -> Result<Self, String> {
        let gpu = GpuBackend::try_new(&data, max_batch_size).await?;
        let model = SmplModel::new(data);
        Ok(GpuSmplModel { model, gpu })
    }

    pub async fn new(data: SmplModelData, max_batch_size: usize) -> Self {
        Self::try_new(data, max_batch_size)
            .await
            .expect("Failed to initialize GPU SMPL model")
    }

    /// Forward vertices using GPU compute shaders.
    ///
    /// Batch size B is inferred from `body_pose.len() / body_pose_len()`.
    /// Panics if B > max_batch_size.
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
        assert!(
            batch_size <= self.gpu.max_batch_size,
            "batch_size {} exceeds max_batch_size {}",
            batch_size, self.gpu.max_batch_size,
        );
        if let Some(pr) = pelvis_rotation {
            assert!(
                pr.len() == batch_size * 3,
                "pelvis_rotation length {} must be exactly batch_size*3 ({})",
                pr.len(),
                batch_size * 3
            );
        }
        if let Some(gr) = global_rotation {
            assert!(
                gr.len() == batch_size * 3,
                "global_rotation length {} must be exactly batch_size*3 ({})",
                gr.len(),
                batch_size * 3
            );
        }
        if let Some(gt) = global_translation {
            assert!(
                gt.len() == batch_size * 3,
                "global_translation length {} must be exactly batch_size*3 ({})",
                gt.len(),
                batch_size * 3
            );
        }

        let s_per = shape.len() / batch_size;
        let s_used = s_per.min(s);

        // === CPU: Joint-level computation for all B instances ===
        let mut shape_full = vec![0.0f32; batch_size * s_used];
        let mut all_pose_delta = Vec::with_capacity(batch_size * p);
        let mut all_joint_rt = Vec::with_capacity(batch_size * j * 12);

        for bi in 0..batch_size {
            let (shape_i, body_pose_i, pelvis_i, global_rot_i, global_trans_i) =
                slice_instance(bi, s_per, bp_len, shape, body_pose, pelvis_rotation, global_rotation, global_translation);

            shape_full[bi * s_used..bi * s_used + s_used].copy_from_slice(&shape_i[..s_used]);

            let jd = self.model.compute_joints(shape_i, body_pose_i, pelvis_i);
            all_pose_delta.extend_from_slice(&compute_pose_delta(&jd.pose_matrices, j, p));

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

        // === GPU: Write into pooled buffers, dispatch 2 passes ===
        let gpu = &self.gpu;
        let b = batch_size;

        // Write dynamic data into pooled buffers
        gpu.queue.write_buffer(&gpu.shape_buf, 0, bytemuck::cast_slice(&shape_full));
        gpu.queue.write_buffer(&gpu.pose_delta_buf, 0, bytemuck::cast_slice(&all_pose_delta));
        gpu.queue.write_buffer(&gpu.joint_rt_buf, 0, bytemuck::cast_slice(&all_joint_rt));

        // Write uniform params
        let blend_params: [u32; 4] = [v as u32, s_used as u32, p as u32, b as u32];
        gpu.queue.write_buffer(&gpu.blend_params_buf, 0, bytemuck::cast_slice(&blend_params));

        let lbs_params: [u32; 4] = [v as u32, d.nnz_per_vertex as u32, j as u32, b as u32];
        gpu.queue.write_buffer(&gpu.lbs_params_buf, 0, bytemuck::cast_slice(&lbs_params));

        // Encode compute passes
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("smpl_forward"),
        });

        let wg_bvd = ((b * v * 3 + 63) / 64) as u32;
        let wg_bv = ((b * v + 63) / 64) as u32;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("shape_pose_blend"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.blend_pipeline);
            pass.set_bind_group(0, &gpu.blend_bg, &[]);
            pass.dispatch_workgroups(wg_bvd, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lbs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu.lbs_pipeline);
            pass.set_bind_group(0, &gpu.lbs_bg, &[]);
            pass.dispatch_workgroups(wg_bv, 1, 1);
        }

        // Copy only the valid portion (not full max_batch_size buffer)
        let copy_bytes = (b * v * 3 * 4) as u64;
        encoder.copy_buffer_to_buffer(&gpu.v_posed_buf, 0, &gpu.staging_buf, 0, copy_bytes);

        gpu.queue.submit(std::iter::once(encoder.finish()));

        // Readback
        let buffer_slice = gpu.staging_buf.slice(..copy_bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        gpu.staging_buf.unmap();

        result
    }
}
