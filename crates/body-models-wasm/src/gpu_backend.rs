//! wgpu device, pipeline, buffer pool, and bind group management for GPU compute.

use wgpu;

use crate::model_data::SmplModelData;

/// GPU resources for SMPL forward pass.
///
/// Pre-allocates all dynamic buffers and bind groups at init time for
/// `max_batch_size`. Per-call overhead is just `write_buffer` + 2 dispatches.
pub struct GpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    // Pipelines (2: fused blend + LBS)
    pub blend_pipeline: wgpu::ComputePipeline,
    pub lbs_pipeline: wgpu::ComputePipeline,

    // Persistent model buffers (uploaded once, read-only)
    pub v_template_buf: wgpu::Buffer,
    pub shapedirs_buf: wgpu::Buffer,
    pub posedirs_buf: wgpu::Buffer,
    pub sparse_indices_buf: wgpu::Buffer,
    pub sparse_weights_buf: wgpu::Buffer,

    // Pooled dynamic buffers (sized for max_batch_size, reused via write_buffer)
    pub shape_buf: wgpu::Buffer,
    pub pose_delta_buf: wgpu::Buffer,
    pub joint_rt_buf: wgpu::Buffer,
    pub v_shaped_buf: wgpu::Buffer,
    pub v_posed_buf: wgpu::Buffer,
    pub staging_buf: wgpu::Buffer,

    // Uniform param buffers (16 bytes each, reused via write_buffer)
    pub blend_params_buf: wgpu::Buffer,
    pub lbs_params_buf: wgpu::Buffer,

    // Pre-created bind groups referencing pooled buffers
    pub blend_bg: wgpu::BindGroup,
    pub lbs_bg: wgpu::BindGroup,

    pub max_batch_size: usize,
}

impl GpuBackend {
    /// Initialize GPU backend with model data and pre-allocated buffer pool.
    pub async fn try_new(model: &SmplModelData, max_batch_size: usize) -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "Failed to find GPU adapter".to_string())?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("SMPL GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            }, None)
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        let v = model.num_vertices;
        let s = model.num_shape_params;
        let p = model.num_pose_params;
        let j = model.num_joints;
        let b = max_batch_size;

        // Create shader modules
        let blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shape_pose_blend"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shape_pose_blend.wgsl").into()),
        });
        let lbs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lbs"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lbs.wgsl").into()),
        });

        // --- Persistent model buffers ---
        let sto = wgpu::BufferUsages::STORAGE;
        let v_template_buf = create_init_buf(&device, "v_template", &model.v_template, sto);
        let shapedirs_buf = create_init_buf(&device, "shapedirs", &model.shapedirs, sto);
        let posedirs_buf = create_init_buf(&device, "posedirs", &model.posedirs, sto);
        let sparse_indices_buf = create_init_buf(&device, "sparse_indices", &model.sparse_indices, sto);
        let sparse_weights_buf = create_init_buf(&device, "sparse_weights", &model.sparse_weights, sto);

        // --- Pooled dynamic buffers ---
        use wgpu::BufferUsages as BU;

        let shape_buf = create_empty_buf(&device, "shape", b * s * 4, BU::STORAGE | BU::COPY_DST);
        let pose_delta_buf = create_empty_buf(&device, "pose_delta", b * p * 4, BU::STORAGE | BU::COPY_DST);
        let joint_rt_buf = create_empty_buf(&device, "joint_rt", b * j * 12 * 4, BU::STORAGE | BU::COPY_DST);
        let v_shaped_buf = create_empty_buf(&device, "v_shaped", b * v * 3 * 4, BU::STORAGE);
        let v_posed_buf = create_empty_buf(&device, "v_posed", b * v * 3 * 4, BU::STORAGE | BU::COPY_SRC);
        let staging_buf = create_empty_buf(&device, "staging", b * v * 3 * 4, BU::MAP_READ | BU::COPY_DST);

        let blend_params_buf = create_empty_buf(&device, "blend_params", 16, BU::UNIFORM | BU::COPY_DST);
        let lbs_params_buf = create_empty_buf(&device, "lbs_params", 16, BU::UNIFORM | BU::COPY_DST);

        // --- Bind group layouts and pipelines ---

        // Fused blend: params, v_template, shapedirs, posedirs, shape, pose_delta, v_shaped
        let blend_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blend_bgl"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(5, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let blend_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blend_pl"),
            bind_group_layouts: &[&blend_bgl],
            push_constant_ranges: &[],
        });
        let blend_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shape_pose_blend"),
            layout: Some(&blend_pl),
            module: &blend_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // LBS: params, v_shaped, sparse_indices, sparse_weights, joint_rt, v_posed
        let lbs_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lbs_bgl"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(5, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let lbs_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lbs_pl"),
            bind_group_layouts: &[&lbs_bgl],
            push_constant_ranges: &[],
        });
        let lbs_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("lbs"),
            layout: Some(&lbs_pl),
            module: &lbs_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Pre-create bind groups ---
        let blend_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blend_bg"),
            layout: &blend_bgl,
            entries: &[
                bg_entry(0, &blend_params_buf),
                bg_entry(1, &v_template_buf),
                bg_entry(2, &shapedirs_buf),
                bg_entry(3, &posedirs_buf),
                bg_entry(4, &shape_buf),
                bg_entry(5, &pose_delta_buf),
                bg_entry(6, &v_shaped_buf),
            ],
        });

        let lbs_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lbs_bg"),
            layout: &lbs_bgl,
            entries: &[
                bg_entry(0, &lbs_params_buf),
                bg_entry(1, &v_shaped_buf),
                bg_entry(2, &sparse_indices_buf),
                bg_entry(3, &sparse_weights_buf),
                bg_entry(4, &joint_rt_buf),
                bg_entry(5, &v_posed_buf),
            ],
        });

        Ok(GpuBackend {
            device,
            queue,
            blend_pipeline,
            lbs_pipeline,
            v_template_buf,
            shapedirs_buf,
            posedirs_buf,
            sparse_indices_buf,
            sparse_weights_buf,
            shape_buf,
            pose_delta_buf,
            joint_rt_buf,
            v_shaped_buf,
            v_posed_buf,
            staging_buf,
            blend_params_buf,
            lbs_params_buf,
            blend_bg,
            lbs_bg,
            max_batch_size,
        })
    }

    /// Initialize GPU backend and panic on failure.
    pub async fn new(model: &SmplModelData, max_batch_size: usize) -> Self {
        Self::try_new(model, max_batch_size)
            .await
            .expect("Failed to initialize GPU backend")
    }
}

pub fn create_init_buf<T: bytemuck::Pod>(device: &wgpu::Device, label: &str, data: &[T], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

fn create_empty_buf(device: &wgpu::Device, label: &str, size_bytes: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size_bytes as u64,
        usage,
        mapped_at_creation: false,
    })
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
