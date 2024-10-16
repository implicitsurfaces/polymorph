// Based on the hello_compute example from the wgpu repo.
// See https://github.com/gfx-rs/wgpu/tree/trunk/examples/src/hello_compute

use bincode;
use bincode::Options;
use fidget::{
    compiler::RegOp,
    context::{Context, Tree},
    var::Var,
    vm::VmData,
};
use std::{borrow::Cow, str::FromStr};
use wgpu::util::DeviceExt;

const GLOBAL_SIZE_X: u32 = 4;
const GLOBAL_SIZE_Y: u32 = 4;
const MAX_TAPE_LEN: usize = 1024;
const OUTPUT_SIZE_BYTES: u32 = GLOBAL_SIZE_X * GLOBAL_SIZE_Y * std::mem::size_of::<f32>() as u32;

include!(concat!(env!("OUT_DIR"), "/opcodes.rs"));

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
                repr[3] = *out;
                repr[7] = *arg;
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
                repr[2] = *out;
                repr[3] = *lhs;
                repr[7] = *rhs;
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
async fn execute_gpu(tape: &[RegOp]) -> Option<Vec<f32>> {
    eprintln!("Executing bytecode: {:?}", tape);

    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await?;

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::SHADER_INT64,
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .unwrap();

    execute_gpu_inner(&device, &queue, tape).await
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tape: &[RegOp],
) -> Option<Vec<f32>> {
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(concat!(
            env!("OUT_DIR"),
            "/shader.wgsl"
        )))),
    });

    let storage_buffer = {
        assert!(tape.len() <= MAX_TAPE_LEN);
        let mut contents = vec![0u8; MAX_TAPE_LEN * std::mem::size_of::<RegOp>()];
        let tape_bytes = tape_to_bytes(tape);
        contents[..tape_bytes.len()].copy_from_slice(&tape_bytes);
        eprintln!("tape bytes {:?}", tape_bytes);

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
        contents: bytemuck::cast_slice(&[GLOBAL_SIZE_X, GLOBAL_SIZE_Y]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: OUTPUT_SIZE_BYTES as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Staging Buffer"),
        size: OUTPUT_SIZE_BYTES as wgpu::BufferAddress,
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

    // A command encoder executes one or many pipelines.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("execute bytecode");
        cpass.dispatch_workgroups(1, 1, 1);
    }
    // Copy the result from the output buffer to the staging buffer
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &output_staging_buffer,
        0,
        OUTPUT_SIZE_BYTES as wgpu::BufferAddress,
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

    #[test]
    fn test_fidget_eval() {
        // From https://docs.rs/fidget/latest/fidget/#functions-and-shapes
        use fidget::{context::Tree, shape::EzShape, vm::VmShape};

        let tree = Tree::x() + Tree::y();
        let shape = VmShape::from(tree);
        let mut eval = VmShape::new_point_eval();
        let tape = shape.ez_point_tape();
        let (out, _) = eval.eval(&tape, 1.0, 1.0, 0.0).unwrap();
        assert_eq!(out, 2.0);
    }

    #[test]
    fn test_fidget_gpu_eval() {
        let tree = Tree::x() + 1;
        let mut ctx = Context::new();
        let sum = ctx.import(&tree);
        let data = VmData::<255>::new(&ctx, &[sum]).unwrap();
        assert_eq!(data.len(), 3); // input, (X + 1), output

        let mut iter = data.iter_asm();
        let vars = &data.vars; // map from var to index
        assert_eq!(iter.next().unwrap(), RegOp::Input(0, vars[&Var::X] as u32));
        assert_eq!(iter.next().unwrap(), RegOp::AddRegImm(0, 0, 1.0));
        assert_eq!(iter.next().unwrap(), RegOp::Output(0, 0));

        let bytecode = data.iter_asm().collect::<Vec<_>>();
        let result = pollster::block_on(execute_gpu(&bytecode));
        assert_eq!(
            result,
            Some(vec![
                1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0
            ])
        );
    }
}
