use bincode;
use fidget::compiler::RegOp;
use std::{borrow::Cow, str::FromStr};
use wgpu::{util::DeviceExt, ShaderStages};

pub mod sdf;

pub const FRAGMENTS_PER_INVOCATION: u32 = 4;
pub const WORKGROUP_SIZE_X: u32 = 16;
pub const WORKGROUP_SIZE_Y: u32 = 16;
pub const MAX_TAPE_LEN_REGOPS: u32 = 32768;
pub const REG_COUNT: usize = 32;

pub fn shader_source() -> String {
    let shared_constants = format!(
        r#"
const WORKGROUP_SIZE_X: u32 = {}u;
const WORKGROUP_SIZE_Y: u32 = {}u;
const MAX_TAPE_LEN_REGOPS: u32 = {}u;
const BYTECODE_ARRAY_LEN: u32 = MAX_TAPE_LEN_REGOPS * 2u;
const REG_COUNT: u32 = {}u;
    "#,
        WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, MAX_TAPE_LEN_REGOPS, REG_COUNT
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
                repr[3] = *out;
                repr[4..8].copy_from_slice(&imm.to_le_bytes());
            }
            RegOp::Load(out, mem) | RegOp::Store(out, mem) => {
                repr[3] = *out;
                repr[4..8].copy_from_slice(&mem.to_le_bytes());
            }
        }
        ans.extend_from_slice(&repr);
    }
    ans
}

#[derive(Copy, Clone)]
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
                required_features: wgpu::Features::SHADER_INT64
                    | wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .unwrap();

    (adapter, device, queue)
}

pub enum Pipeline<'a> {
    Compute(&'a wgpu::ComputePipeline),
    Render(&'a wgpu::RenderPipeline),
}

pub struct Buffers {
    pub bytecode_buffer: wgpu::Buffer,
    pub pc_max_buffer: wgpu::Buffer,
    pub output_buffer: wgpu::Buffer,
    pub output_staging_buffer: wgpu::Buffer,
    pub dims_buffer: wgpu::Buffer,
    pub timestamp_resolve_buffer: wgpu::Buffer,
    pub timestamp_readback_buffer: wgpu::Buffer,
}

pub fn create_buffers(
    device: &wgpu::Device,
    tape: &[RegOp],
    invoc_size: (u32, u32),
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
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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

    let dims_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dimensions Buffer"),
        contents: bytemuck::cast_slice(&[invoc_size.0, invoc_size.1]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create two buffers for timestamps
    let timestamp_resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Timestamp Resolve Buffer"),
        size: 16, // 2 u64 values
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
        mapped_at_creation: false,
    });

    let timestamp_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Timestamp Readback Buffer"),
        size: 16, // 2 u64 values
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    Buffers {
        bytecode_buffer,
        pc_max_buffer,
        output_buffer,
        output_staging_buffer,
        dims_buffer,
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
        ],
    })
}

pub fn setup_pipeline_layout(
    device: &wgpu::Device,
    shader_stages: ShaderStages,
) -> wgpu::PipelineLayout {
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
        ],
    });

    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    })
}
