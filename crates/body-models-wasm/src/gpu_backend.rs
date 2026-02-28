//! wgpu device, pipeline, and buffer management for GPU compute.

use wgpu;

use crate::model_data::SmplModelData;

/// GPU resources for SMPL forward pass.
///
/// Stores device, pipelines, static model buffers, and bind group layouts.
/// Dynamic buffers and bind groups are created per-call in gpu_forward.rs
/// to support arbitrary batch sizes.
pub struct GpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,

    // Pipelines
    pub shape_blend_pipeline: wgpu::ComputePipeline,
    pub pose_blend_pipeline: wgpu::ComputePipeline,
    pub lbs_pipeline: wgpu::ComputePipeline,

    // Bind group layouts (needed for per-call bind group creation)
    pub shape_blend_bgl: wgpu::BindGroupLayout,
    pub pose_blend_bgl: wgpu::BindGroupLayout,
    pub lbs_bgl: wgpu::BindGroupLayout,

    // Persistent model buffers (uploaded once, read-only)
    pub v_template_buf: wgpu::Buffer,
    pub shapedirs_buf: wgpu::Buffer,
    pub posedirs_buf: wgpu::Buffer,
    pub sparse_indices_buf: wgpu::Buffer,
    pub sparse_weights_buf: wgpu::Buffer,
}

impl GpuBackend {
    /// Initialize GPU backend with model data. Must be called with an async runtime.
    pub async fn new(model: &SmplModelData) -> Self {
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
            .expect("Failed to find GPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("SMPL GPU"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            }, None)
            .await
            .expect("Failed to create device");

        // Create shader modules
        let shape_blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shape_blend"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shape_blend.wgsl").into()),
        });
        let pose_blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pose_blend"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/pose_blend.wgsl").into()),
        });
        let lbs_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lbs"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/lbs.wgsl").into()),
        });

        // --- Create persistent model buffers ---
        let sto = wgpu::BufferUsages::STORAGE;
        let v_template_buf = create_init_buf(&device, "v_template", &model.v_template, sto);
        let shapedirs_buf = create_init_buf(&device, "shapedirs", &model.shapedirs, sto);
        let posedirs_buf = create_init_buf(&device, "posedirs", &model.posedirs, sto);
        let sparse_indices_buf = create_init_buf(&device, "sparse_indices", &model.sparse_indices, sto);
        let sparse_weights_buf = create_init_buf(&device, "sparse_weights", &model.sparse_weights, sto);

        // --- Create bind group layouts and pipelines ---

        // Shape blend: params, v_template, shapedirs, shape, v_shaped
        let shape_blend_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shape_blend_bgl"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let shape_blend_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shape_blend_pl"),
            bind_group_layouts: &[&shape_blend_bgl],
            push_constant_ranges: &[],
        });
        let shape_blend_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("shape_blend"),
            layout: Some(&shape_blend_pl),
            module: &shape_blend_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Pose blend: params, posedirs, pose_delta, v_shaped
        let pose_blend_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pose_blend_bgl"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pose_blend_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pose_blend_pl"),
            bind_group_layouts: &[&pose_blend_bgl],
            push_constant_ranges: &[],
        });
        let pose_blend_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pose_blend"),
            layout: Some(&pose_blend_pl),
            module: &pose_blend_shader,
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

        GpuBackend {
            device,
            queue,
            shape_blend_pipeline,
            pose_blend_pipeline,
            lbs_pipeline,
            shape_blend_bgl,
            pose_blend_bgl,
            lbs_bgl,
            v_template_buf,
            shapedirs_buf,
            posedirs_buf,
            sparse_indices_buf,
            sparse_weights_buf,
        }
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

pub fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}
