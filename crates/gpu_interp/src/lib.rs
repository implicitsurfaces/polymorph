use bincode;
use fidget::{
    compiler::RegOp,
    shape::EzShape,
    vm::{VmFunction, VmShape},
};

use wgpu::{util::DeviceExt, ShaderStages};

pub mod sdf;

pub const FRAGMENTS_PER_INVOCATION: u32 = 4;
pub const TIMESTAMP_COUNT: u64 = 4;
pub const WORKGROUP_SIZE_X: u32 = 16;
pub const WORKGROUP_SIZE_Y: u32 = 16;
pub const MAX_TAPE_LEN_REGOPS: u32 = 32768 * 5;
pub const REG_COUNT: usize = 255;
pub const TILE_SIZE_X: u32 = 64;
pub const TILE_SIZE_Y: u32 = 32;
pub const MAX_TILE_COUNT: u32 = 256;

pub fn shader_source() -> String {
    let shared_constants = format!(
        r#"
const WORKGROUP_SIZE_X: u32 = {WORKGROUP_SIZE_X}u;
const WORKGROUP_SIZE_Y: u32 = {WORKGROUP_SIZE_Y}u;
const MAX_TAPE_LEN_REGOPS: u32 = {MAX_TAPE_LEN_REGOPS}u;
const BYTECODE_ARRAY_LEN: u32 = MAX_TAPE_LEN_REGOPS * 2u;
const REG_COUNT: u32 = {REG_COUNT}u;
const TILE_SIZE_X: u32 = {TILE_SIZE_X}u;
const TILE_SIZE_Y: u32 = {TILE_SIZE_Y}u;
const MAX_TILE_COUNT: u32 = {MAX_TILE_COUNT}u;
const MAX_TILE_COUNT_DIV_4: u32 = MAX_TILE_COUNT / 4u;
    "#
    );
    include_str!("shader-in.wgsl")
        .to_string()
        .replace("{ shared_constants }", shared_constants.as_ref())
}

// TODO: Find a better way to do  this.
pub fn regops_remapping_vars(
    point_tape: &fidget::shape::ShapeTape<fidget::vm::GenericVmFunction<255>>,
) -> Vec<RegOp> {
    let data = point_tape.raw_tape().data();
    let varmap = point_tape.vars();
    let mut remapping = [None; 3];

    if let Some(idx) = varmap.get(&fidget::var::Var::X) {
        remapping[idx] = Some(0);
    }
    if let Some(idx) = varmap.get(&fidget::var::Var::Y) {
        remapping[idx] = Some(1);
    }
    if let Some(idx) = varmap.get(&fidget::var::Var::Z) {
        remapping[idx] = Some(2);
    }

    data.iter_asm()
        .map(|r| match r {
            RegOp::Input(out, i) => RegOp::Input(out, remapping[i as usize].unwrap()),
            other => other,
        })
        .collect()
}

pub struct GPUTape {
    pub tape: Vec<RegOp>,
    pub offsets: Vec<u32>,
    pub lengths: Vec<u32>,
}

impl GPUTape {
    pub fn new(ctx: fidget::Context, root: fidget::context::Node, width: u32, height: u32) -> Self {
        let start = std::time::Instant::now();

        let shape = VmShape::new(&ctx, root).unwrap();

        // The default (unsimplified) tape is always the first one we right.
        // But not that it's *not* accounted for in `subtape_starts` and
        // `subtape_ends` — those are only for looking up the simplified
        // tapes.
        // TODO: Find a cleaner way to do this?
        let default_tape = shape.ez_point_tape();
        let mut regops: Vec<RegOp> = regops_remapping_vars(&default_tape);
        let default_tape_len = regops.len() as u32;

        let mut subtape_starts: Vec<u32> = Vec::new();
        let mut subtape_ends: Vec<u32> = Vec::new();

        // tiling
        let ret = {
            let tape_i = shape.ez_interval_tape();
            let mut eval_i = fidget::shape::Shape::<VmFunction>::new_interval_eval();

            let unproject = |val: f32, bounds: f32| (2.0 * val / bounds) - 1.0;

            for row in 0..(height / TILE_SIZE_Y) {
                let y = row as f32 * TILE_SIZE_Y as f32;
                let y_interval = fidget::types::Interval::new(
                    -unproject(y + TILE_SIZE_Y as f32, height as f32),
                    -unproject(y, height as f32),
                );

                for col in 0..(width / TILE_SIZE_X) {
                    let x = col as f32 * TILE_SIZE_X as f32;
                    let x_interval = fidget::types::Interval::new(
                        unproject(x, width as f32),
                        unproject(x + TILE_SIZE_X as f32, width as f32),
                    );
                    dbg!(x_interval, y_interval);

                    let (out, trace) = eval_i
                        .eval(&tape_i, x_interval, y_interval, 0.0.into())
                        .unwrap();
                    if out.lower() > 0.0 {
                        // The entire tile is outside the shape -- write an empty tape.
                        subtape_starts.push(regops.len() as u32 * 2);
                        subtape_ends.push(regops.len() as u32 * 2);
                        continue;
                    }
                    if out.upper() < 0.0 || trace.is_none() {
                        // Tile is entirely inside the shape, or the tape could not
                        // be simplified. Use the default tape.
                        // TODO: Could we do better in the "entirely inside" case?
                        subtape_starts.push(0);
                        subtape_ends.push(default_tape_len * 2);
                        continue;
                    }
                    let simplified_tape =
                        shape.ez_simplify(trace.unwrap()).unwrap().ez_point_tape();
                    subtape_starts.push(regops.len() as u32 * 2);
                    regops.extend(regops_remapping_vars(&simplified_tape));
                    subtape_ends.push(regops.len() as u32 * 2);
                }
            }
            dbg!(&subtape_starts);
            dbg!(&subtape_ends);
            // dbg!(&regops);
            GPUTape {
                tape: regops,
                offsets: dbg!(subtape_starts),
                lengths: dbg!(subtape_ends),
            }
        };

        let duration = start.elapsed();
        eprintln!("GPUTape::new took {:?}", duration);

        ret
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut ans: Vec<u8> = Vec::new();
        for op in &self.tape {
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
                RegOp::Load(_r, _mem) | RegOp::Store(_r, _mem) => {
                    panic!("Store/load not supported. Try increasing REG_COUNT.")
                }
            }
            ans.extend_from_slice(&repr);
        }
        ans
    }
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

#[derive(Copy, Clone, Debug)]
pub struct Projection {
    pub scale: [f32; 2],
    pub translation: [f32; 2],
}

impl Projection {
    pub fn as_bytes(&self) -> [u8; 4 * std::mem::size_of::<f32>()] {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&self.scale[0].to_le_bytes());
        bytes[4..8].copy_from_slice(&self.scale[1].to_le_bytes());
        bytes[8..12].copy_from_slice(&self.translation[0].to_le_bytes());
        bytes[12..16].copy_from_slice(&self.translation[1].to_le_bytes());
        bytes
    }
}
impl Default for Projection {
    fn default() -> Self {
        Self {
            scale: [1., 1.],
            translation: [0., 0.],
        }
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
    pub offsets_buffer: wgpu::Buffer,
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
    tape: &GPUTape,
    viewport: Viewport,
    projection: Projection,
) -> Buffers {
    let bytecode_buffer = {
        let mut contents = vec![0u8; MAX_TAPE_LEN_REGOPS as usize * std::mem::size_of::<RegOp>()];
        let tape_bytes = tape.to_bytes();
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

    let tile_count = (viewport.width / TILE_SIZE_X) * (viewport.height / TILE_SIZE_Y);
    assert!(viewport.width % TILE_SIZE_X == 0);
    assert!(viewport.height % TILE_SIZE_X == 0);
    assert!(tape.lengths.len() == tile_count as usize);

    let offsets_buffer = {
        let mut contents = vec![0u32; MAX_TILE_COUNT as usize];
        contents[..tape.offsets.len()].copy_from_slice(&tape.offsets);
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Offsets Buffer"),
            contents: bytemuck::cast_slice(&contents),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    };

    let pc_max_buffer = {
        let mut contents = vec![0u32; MAX_TILE_COUNT as usize];
        contents[..tape.lengths.len()].copy_from_slice(&tape.lengths);
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pc_max Buffer"),
            contents: bytemuck::cast_slice(&contents),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    };

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
        contents: { bytemuck::cast_slice(&projection.as_bytes()) },
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
        offsets_buffer,
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
                resource: buffers.offsets_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.pc_max_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffers.output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buffers.dims_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buffers.step_count_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
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
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: shader_stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            wgpu::BindGroupLayoutEntry {
                binding: 6,
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
        // eprintln!(
        //     "Duration #1: {:?}",
        //     Duration::from_nanos((p * (timestamps[1] - timestamps[0]) as f32) as u64)
        // );
        // eprintln!(
        //     "Duration #2: {:?}",
        //     Duration::from_nanos((p * (timestamps[3] - timestamps[2]) as f32) as u64)
        // );
        drop(timestamp_data);
        buffers.timestamp_readback_buffer.unmap();
    }
}

pub fn add_compute_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    pipeline: &'a wgpu::ComputePipeline,
    bind_group: &'a wgpu::BindGroup,
    timestamp_query_set: Option<&'a wgpu::QuerySet>,
    viewport: &Viewport,
) {
    let invoc_size = (viewport.width / FRAGMENTS_PER_INVOCATION, viewport.height);
    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: timestamp_query_set.map(|query_set| wgpu::ComputePassTimestampWrites {
            query_set,
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
