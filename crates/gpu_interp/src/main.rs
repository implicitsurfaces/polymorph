// Based on the hello_compute example from the wgpu repo.
// See https://github.com/gfx-rs/wgpu/tree/trunk/examples/src/hello_compute

use gpu_interp::*;

use bincode;
use fidget::compiler::RegOp;
use std::{borrow::Cow, str::FromStr};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE_X: u32 = 16;
const WORKGROUP_SIZE_Y: u32 = 16;
const MAX_TAPE_LEN_REGOPS: u32 = 32768;
const REG_COUNT: usize = 32;

fn shader_source() -> String {
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

fn tape_to_bytes(tape: &[RegOp]) -> Vec<u8> {
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

#[cfg_attr(test, allow(dead_code))]
async fn run() {
    let numbers = if std::env::args().len() <= 2 {
        let default = vec![4, 4, 4, 4, 1];
        println!("No bytecode was provided, defaulting to {default:?}");
        default
    } else {
        std::env::args()
            .skip(2)
            .map(|s| u32::from_str(&s).expect("You must pass a list of positive integers!"))
            .collect()
    };

    // let result = execute_gpu(&numbers).await.unwrap();

    println!("Output: {:?}", numbers);
}

#[cfg_attr(test, allow(dead_code))]
async fn execute_gpu(tape: &[RegOp], data_size: (u32, u32)) -> Option<Vec<f32>> {
    // eprintln!("Executing bytecode: {:?}", tape);

    let instance = wgpu::Instance::default();
    let options = wgpu::RequestAdapterOptions::default();
    let (_, device, queue) = create_device(&instance, &options).await;
    execute_gpu_inner(&device, &queue, tape, data_size).await
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tape: &[RegOp],
    data_size: (u32, u32),
) -> Option<Vec<f32>> {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source())),
    });
    let invoc_size = (data_size.0 / 4, data_size.1);

    let storage_buffer = {
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
            label: Some("Storage Buffer"),
            contents: &contents,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::cast_slice(&[tape.len() as u32 * 2]), // x2 because each instruction is 2 u32s
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let dimensions_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dimensions Buffer"),
        contents: bytemuck::cast_slice(&[invoc_size.0, invoc_size.1]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let output_size_bytes = data_size.0 * data_size.1 * std::mem::size_of::<f32>() as u32;

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

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "compute_main",
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dimensions_buffer.as_entire_binding(),
            },
        ],
    });

    // Create timestamp query set
    let timestamp_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("Timestamp Query Set"),
        count: 2,
        ty: wgpu::QueryType::Timestamp,
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

    // A command encoder executes one or many pipelines.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &timestamp_query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        cpass.set_pipeline(&compute_pipeline);
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
    // Copy the result from the output buffer to the staging buffer
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &output_staging_buffer,
        0,
        output_size_bytes as wgpu::BufferAddress,
    );

    // Resolve timestamp query results
    encoder.resolve_query_set(&timestamp_query_set, 0..2, &timestamp_resolve_buffer, 0);

    // Copy timestamp results from resolve buffer to readback buffer
    encoder.copy_buffer_to_buffer(
        &timestamp_resolve_buffer,
        0,
        &timestamp_readback_buffer,
        0,
        16,
    );

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Sets up staging buffer for mapping, sending the result back when finished.
    let buffer_slice = output_staging_buffer.slice(..);
    let (sender, receiver) = flume::bounded(1);
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    device.poll(wgpu::Maintain::wait()).panic_on_timeout();

    // Awaits until `buffer_future` can be read from
    if let Ok(Ok(())) = receiver.recv_async().await {
        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();
        // Convert contents to f32
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        // Unmap buffer
        drop(data);
        output_staging_buffer.unmap();

        // Map the timestamp readback buffer and read the results
        let timestamp_slice = timestamp_readback_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        timestamp_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        if let Ok(Ok(())) = receiver.recv_async().await {
            let timestamp_data = timestamp_slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&timestamp_data);
            let duration_ns = timestamps[1] - timestamps[0];
            eprintln!(
                "GPU execution time: {} ms",
                duration_ns as f32 / 1_000_000.0
            );
            drop(timestamp_data);
            timestamp_readback_buffer.unmap();
        }

        // Returns data from buffer
        Some(result)
    } else {
        panic!("failed to run compute on gpu!")
    }
}

pub fn main() {
    env_logger::init();
    pollster::block_on(run());
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_relative_eq;
    use fidget::{
        compiler::RegOp,
        context::{Context, Tree},
        jit::JitShape,
        shape::EzShape,
        var::Var,
        vm::VmData,
    };

    #[test]
    fn test_fidget_gpu_eval() {
        let tree = Tree::x() + 1;
        let mut ctx = Context::new();
        let sum = ctx.import(&tree);
        let data = VmData::<REG_COUNT>::new(&ctx, &[sum]).unwrap();
        assert_eq!(data.len(), 3); // input, (X + 1), output

        let mut iter = data.iter_asm();
        let vars = &data.vars; // map from var to index
        assert_eq!(iter.next().unwrap(), RegOp::Input(0, vars[&Var::X] as u32));
        assert_eq!(iter.next().unwrap(), RegOp::AddRegImm(0, 0, 1.0));
        assert_eq!(iter.next().unwrap(), RegOp::Output(0, 0));

        let data_size = (64, 16);
        let bytecode = data.iter_asm().collect::<Vec<_>>();
        let result = pollster::block_on(execute_gpu(&bytecode, data_size));
        assert_eq!(result.unwrap(), jit_evaluate(&tree, data_size));
    }

    #[test]
    fn test_fidget_four_circles() {
        let mut circles = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                let center_x = i as f64;
                let center_y = j as f64;
                circles.push(circle(center_x, center_y, 0.5));
            }
        }
        let tree = smooth_union(circles);
        let mut ctx = Context::new();
        let node = ctx.import(&tree);
        let data = VmData::<REG_COUNT>::new(&ctx, &[node]).unwrap();
        // debug!("{:?}", data.iter_asm().collect::<Vec<_>>());

        let data_size = (64, 64);
        let bytecode = data.iter_asm().collect::<Vec<_>>();
        // eprintln!("{:?}", bytecode);
        let result = pollster::block_on(execute_gpu(&bytecode, data_size));
        assert_relative_eq!(
            result.unwrap().as_slice(),
            jit_evaluate(&tree, data_size).as_slice(),
            epsilon = 1e-1
        );
    }

    fn smooth_union(trees: Vec<Tree>) -> Tree {
        trees
            .into_iter()
            .reduce(|a, b| {
                let k = 0.1;
                let k_doubled = k * 2.0;
                let x = b.clone() - a.clone();
                0.5 * (a + b - (x.square() + k_doubled * k_doubled).sqrt())
            })
            .unwrap()
    }

    #[test]
    fn test_fidget_many_circles() {
        let mut circles = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                let center_x = i as f64;
                let center_y = j as f64;
                circles.push(circle(center_x, center_y, 0.5));
            }
        }
        let tree = smooth_union(circles);
        let start = std::time::Instant::now();
        let mut ctx = Context::new();
        let node = ctx.import(&tree);
        let duration = start.elapsed();
        let data = VmData::<REG_COUNT>::new(&ctx, &[node]).unwrap();

        eprintln!("Bytecode compilation took {:?}", duration);

        let data_size = (512, 512);
        let bytecode = data.iter_asm().collect::<Vec<_>>();
        // debug!("{:?}", bytecode);
        let result = pollster::block_on(execute_gpu(&bytecode, data_size));
        assert_relative_eq!(
            result.unwrap().as_slice(),
            jit_evaluate(&tree, data_size).as_slice(),
            epsilon = 1.0
        );
    }

    fn circle(center_x: f64, center_y: f64, radius: f64) -> Tree {
        let dx = Tree::constant(center_x) - Tree::x();
        let dy = Tree::constant(center_y) - Tree::y();
        let dist = (dx.square() + dy.square()).sqrt();
        return dist - radius;
    }

    fn grid_sample(
        x_max: f32,
        y_max: f32,
        x_steps: u32,
        y_steps: u32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut x = Vec::with_capacity(x_steps as usize * y_steps as usize);
        let mut y = Vec::with_capacity(x_steps as usize * y_steps as usize);
        let mut z = Vec::with_capacity(x_steps as usize * y_steps as usize);

        let x_step = x_max / (x_steps - 1) as f32;
        let y_step = y_max / (y_steps - 1) as f32;

        for i in 0..y_steps {
            for j in 0..x_steps {
                let x_val = j as f32 * x_step;
                let y_val = i as f32 * y_step;

                x.push(x_val);
                y.push(y_val);
                z.push(0.0);
            }
        }

        (x, y, z)
    }

    fn jit_evaluate(tree: &Tree, data_size: (u32, u32)) -> Vec<f32> {
        let shape = JitShape::from(tree.clone());
        let tape = shape.ez_float_slice_tape();
        let mut eval = JitShape::new_float_slice_eval();

        let (x, y, z) = grid_sample(
            data_size.0 as f32 - 1.0,
            data_size.1 as f32 - 1.0,
            data_size.0,
            data_size.1,
        );

        let start = std::time::Instant::now();
        let _ = eval.eval(&tape, x.as_slice(), y.as_slice(), z.as_slice());
        eprintln!("Jit eval took {:?}", start.elapsed());

        let start = std::time::Instant::now();
        let r = eval.eval(&tape, x.as_slice(), y.as_slice(), z.as_slice());
        eprintln!("Jit eval #2 took {:?}", start.elapsed());
        r.unwrap().to_vec()
    }
}
