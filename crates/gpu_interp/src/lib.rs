use bincode;
use fidget::compiler::RegOp;
use std::time::Duration;
use wgpu::{util::DeviceExt, ShaderStages};

pub mod sdf;

pub const FRAGMENTS_PER_INVOCATION: u32 = 4;
pub const TIMESTAMP_COUNT: u64 = 4;
pub const WORKGROUP_SIZE_X: u32 = 16;
pub const WORKGROUP_SIZE_Y: u32 = 16;
pub const MAX_TAPE_LEN_REGOPS: u32 = 32768;
pub const REG_COUNT: usize = 32;
pub const MEM_SIZE: usize = 32;

pub fn shader_source() -> String {
    let shared_constants = format!(
        r#"
const WORKGROUP_SIZE_X: u32 = {WORKGROUP_SIZE_X}u;
const WORKGROUP_SIZE_Y: u32 = {WORKGROUP_SIZE_Y}u;
const MAX_TAPE_LEN_REGOPS: u32 = {MAX_TAPE_LEN_REGOPS}u;
const BYTECODE_ARRAY_LEN: u32 = MAX_TAPE_LEN_REGOPS * 2u;
const REG_COUNT: u32 = {REG_COUNT}u;
const MEM_SIZE: u32 = {MEM_SIZE}u;
    "#
    );
    include_str!("shader-in.wgsl")
        .to_string()
        .replace("{ shared_constants }", shared_constants.as_ref())
}

pub fn tape_to_bytes(tape: &[RegOp]) -> Vec<u8> {
    let mut ans: Vec<u8> = Vec::new();
    for op in tape {
        // This is very naughty! bincode will serialize the enum discriminant
        // as a u32, but we know that the discriminant is always a single byte.
        let tag = bincode::serialize(op).unwrap()[0];
        let mut repr = [0u8; 8];
        repr[0] = tag;
        match op {
            RegOp::Input(out, i) => {
                repr[1] = *out;
                repr[4..8].copy_from_slice(&i.to_le_bytes());
            }
            RegOp::Output(arg, i) => {
                repr[1] = *arg;
                repr[4..8].copy_from_slice(&i.to_le_bytes());
            }
            RegOp::NegReg(out, arg)
            | RegOp::AbsReg(out, arg)
            | RegOp::RecipReg(out, arg)
            | RegOp::SqrtReg(out, arg)
            | RegOp::SquareReg(out, arg)
            | RegOp::FloorReg(out, arg)
            | RegOp::CeilReg(out, arg)
            | RegOp::RoundReg(out, arg)
            | RegOp::SinReg(out, arg)
            | RegOp::CosReg(out, arg)
            | RegOp::TanReg(out, arg)
            | RegOp::AsinReg(out, arg)
            | RegOp::AcosReg(out, arg)
            | RegOp::AtanReg(out, arg)
            | RegOp::ExpReg(out, arg)
            | RegOp::LnReg(out, arg)
            | RegOp::NotReg(out, arg)
            | RegOp::CopyReg(out, arg) => {
                repr[1] = *out;
                repr[2] = *arg;
            }
            RegOp::AddRegImm(out, arg, imm)
            | RegOp::MulRegImm(out, arg, imm)
            | RegOp::DivRegImm(out, arg, imm)
            | RegOp::DivImmReg(out, arg, imm)
            | RegOp::AtanRegImm(out, arg, imm)
            | RegOp::AtanImmReg(out, arg, imm)
            | RegOp::SubImmReg(out, arg, imm)
            | RegOp::SubRegImm(out, arg, imm)
            | RegOp::MinRegImm(out, arg, imm)
            | RegOp::MaxRegImm(out, arg, imm)
            | RegOp::AndRegImm(out, arg, imm)
            | RegOp::OrRegImm(out, arg, imm)
            | RegOp::ModRegImm(out, arg, imm)
            | RegOp::ModImmReg(out, arg, imm)
            | RegOp::CompareRegImm(out, arg, imm)
            | RegOp::CompareImmReg(out, arg, imm) => {
                repr[1] = *out;
                repr[2] = *arg;
                repr[4..8].copy_from_slice(&imm.to_le_bytes());
            }
            RegOp::AtanRegReg(out, lhs, rhs)
            | RegOp::AndRegReg(out, lhs, rhs)
            | RegOp::OrRegReg(out, lhs, rhs)
            | RegOp::ModRegReg(out, lhs, rhs)
            | RegOp::AddRegReg(out, lhs, rhs)
            | RegOp::MulRegReg(out, lhs, rhs)
            | RegOp::DivRegReg(out, lhs, rhs)
            | RegOp::SubRegReg(out, lhs, rhs)
            | RegOp::CompareRegReg(out, lhs, rhs)
            | RegOp::MinRegReg(out, lhs, rhs)
            | RegOp::MaxRegReg(out, lhs, rhs) => {
                repr[1] = *out;
                repr[2] = *lhs;
                repr[3] = *rhs;
            }
            RegOp::CopyImm(out, imm) => {
                repr[1] = *out;
                repr[4..8].copy_from_slice(&imm.to_le_bytes());
            }
            RegOp::Load(r, mem) | RegOp::Store(r, mem) => {
                repr[1] = *r;
                repr[4..8].copy_from_slice(&mem.to_le_bytes());
            }
        }
        ans.extend_from_slice(&repr);
    }
    ans
}

#[derive(Copy, Clone, Debug)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    pub fn byte_size(&self) -> u32 {
        self.width * self.height * std::mem::size_of::<f32>() as u32
    }
}

pub async fn create_device(
    instance: &wgpu::Instance,
    options: &wgpu::RequestAdapterOptions<'_, '_>,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance.request_adapter(options).await.unwrap();

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::SHADER_INT64 | wgpu::Features::TIMESTAMP_QUERY,
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .unwrap();

    (adapter, device, queue)
}

pub struct Buffers {
    pub bytecode_buffer: wgpu::Buffer,
    pub pc_max_buffer: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    pub output_staging_buffer: wgpu::Buffer,
    pub dims_buffer: wgpu::Buffer,
    pub step_count_buffer: wgpu::Buffer,
    pub projection_buffer: wgpu::Buffer,
    pub timestamp_resolve_buffer: wgpu::Buffer,
    pub timestamp_readback_buffer: wgpu::Buffer,
}

pub fn create_and_fill_buffers(
    device: &wgpu::Device,
    tape: &[RegOp],
    viewport: Viewport,
) -> Buffers {
    let bytecode_buffer = {
        assert!(
            tape.len() <= MAX_TAPE_LEN_REGOPS as usize,
            "Tape too long: {:?}",
            tape.len()
        );
        let mut contents = vec![0u8; MAX_TAPE_LEN_REGOPS as usize * std::mem::size_of::<RegOp>()];
        let tape_bytes = tape_to_bytes(tape);
        contents[..tape_bytes.len()].copy_from_slice(&tape_bytes);
        // eprintln!("tape bytes {:?}", tape_bytes);

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bytecode Buffer"),
            contents: &contents,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    };

    let pc_max_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pc_max Buffer"),
        contents: bytemuck::cast_slice(&[tape.len() as u32 * 2]), // x2 because each instruction is 2 u32s
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let output_size_bytes = viewport.byte_size();

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_size_bytes as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Staging Buffer"),
        size: output_size_bytes as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let viewport_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Viewport Buffer"),
        contents: bytemuck::cast_slice(&[viewport.width, viewport.height]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let step_count_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Step Count Buffer"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let projection_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Projection Buffer"),
        // maps screen space to world space: scale, then translate
        contents: {
            let w = viewport.width as f32;
            let h = viewport.height as f32;
            bytemuck::cast_slice(&dbg!([
                1. / (w / 2.) as f32,
                -1. / (h / 2.) as f32,
                1.,
                -1.
            ]))
        },
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create two buffers for timestamps
    let timestamp_resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Timestamp Resolve Buffer"),
        size: 4 * std::mem::size_of::<u64>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });

    let timestamp_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Timestamp Readback Buffer"),
        size: 4 * std::mem::size_of::<u64>() as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    Buffers {
        bytecode_buffer,
        pc_max_buffer,
        output_buffer,
        output_staging_buffer,
        dims_buffer: viewport_buffer,
        step_count_buffer,
        projection_buffer,
        timestamp_resolve_buffer,
        timestamp_readback_buffer,
    }
}

pub fn create_bind_group(
    device: &wgpu::Device,
    buffers: &Buffers,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.bytecode_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.pc_max_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffers.dims_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffers.step_count_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buffers.projection_buffer.as_entire_binding(),
            },
        ],
    })
}

pub fn setup_pipeline_layout(
    device: &wgpu::Device,
    shader_stages: ShaderStages,
) -> (wgpu::PipelineLayout, wgpu::BindGroupLayout) {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: shader_stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: shader_stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: shader_stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: shader_stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: shader_stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: shader_stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    (pipeline_layout, bind_group_layout)
}

pub async fn print_timestamps(device: &wgpu::Device, queue: &wgpu::Queue, buffers: &Buffers) {
    // Map the timestamp readback buffer and read the results
    let timestamp_slice = buffers.timestamp_readback_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    timestamp_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    let p = queue.get_timestamp_period();

    if let Ok(Ok(())) = receiver.recv_async().await {
        let timestamp_data = timestamp_slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&timestamp_data);
        eprintln!(
            "Duration #1: {:?}",
            Duration::from_nanos((p * (timestamps[1] - timestamps[0]) as f32) as u64)
        );
        eprintln!(
            "Duration #2: {:?}",
            Duration::from_nanos((p * (timestamps[3] - timestamps[2]) as f32) as u64)
        );
        drop(timestamp_data);
        buffers.timestamp_readback_buffer.unmap();
    }
}

pub fn add_compute_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    pipeline: &'a wgpu::ComputePipeline,
    bind_group: &'a wgpu::BindGroup,
    timestamp_query_set: &'a wgpu::QuerySet,
    viewport: &Viewport,
) {
    let invoc_size = (viewport.width / FRAGMENTS_PER_INVOCATION, viewport.height);
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
            query_set: &timestamp_query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        }),
    });
    cpass.set_pipeline(pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    cpass.insert_debug_marker("execute bytecode");
    assert!(invoc_size.0 % WORKGROUP_SIZE_X == 0);
    assert!(invoc_size.1 % WORKGROUP_SIZE_Y == 0);
    cpass.dispatch_workgroups(
        invoc_size.0 / WORKGROUP_SIZE_X,
        invoc_size.1 / WORKGROUP_SIZE_Y,
        1,
    );
}
