// Based on the hello_compute example from the wgpu repo.
// See https://github.com/gfx-rs/wgpu/tree/trunk/examples/src/hello_compute

use gpu_interp::*;

use fidget::compiler::RegOp;
use std::{borrow::Cow, str::FromStr};

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

    // let result = evaluate_tape(&numbers).await.unwrap();

    println!("Output: {:?}", numbers);
}

#[allow(dead_code)]
async fn evaluate_tape(tape: &[RegOp], viewport: Viewport) -> Option<Vec<f32>> {
    let (_, device, queue) = create_device(
        &wgpu::Instance::default(),
        &wgpu::RequestAdapterOptions::default(),
    )
    .await;

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source())),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&setup_pipeline_layout(&device, wgpu::ShaderStages::COMPUTE)),
        module: &shader_module,
        entry_point: "compute_main",
        compilation_options: Default::default(),
        cache: None,
    });

    let buffers = create_buffers(&device, tape, viewport);
    let bind_group = create_bind_group(&device, &buffers, &pipeline.get_bind_group_layout(0));

    // Create timestamp query set
    let timestamp_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("Timestamp Query Set"),
        count: 2,
        ty: wgpu::QueryType::Timestamp,
    });

    // A command encoder executes one or many pipelines.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let invoc_size = (viewport.width / FRAGMENTS_PER_INVOCATION, viewport.height);
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &timestamp_query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        cpass.set_pipeline(&pipeline);
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
        &buffers.output_buffer,
        0,
        &buffers.output_staging_buffer,
        0,
        viewport.byte_size() as wgpu::BufferAddress,
    );

    // Resolve timestamp query results
    encoder.resolve_query_set(
        &timestamp_query_set,
        0..2,
        &buffers.timestamp_resolve_buffer,
        0,
    );

    // Copy timestamp results from resolve buffer to readback buffer
    encoder.copy_buffer_to_buffer(
        &buffers.timestamp_resolve_buffer,
        0,
        &buffers.timestamp_readback_buffer,
        0,
        16,
    );

    // Submits command encoder for processing
    queue.submit(Some(encoder.finish()));

    // Sets up staging buffer for mapping, sending the result back when finished.
    let buffer_slice = buffers.output_staging_buffer.slice(..);
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
        buffers.output_staging_buffer.unmap();

        print_timestamps(&device, &buffers).await;

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
    use sdf::*;

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

        let viewport = Viewport {
            width: 64,
            height: 16,
        };
        let bytecode = data.iter_asm().collect::<Vec<_>>();
        let result = pollster::block_on(evaluate_tape(&bytecode, viewport));
        assert_eq!(result.unwrap(), jit_evaluate(&tree, viewport));
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

        let viewport = Viewport {
            width: 64,
            height: 64,
        };
        let bytecode = data.iter_asm().collect::<Vec<_>>();
        // eprintln!("{:?}", bytecode);
        let result = pollster::block_on(evaluate_tape(&bytecode, viewport));
        assert_relative_eq!(
            result.unwrap().as_slice(),
            jit_evaluate(&tree, viewport).as_slice(),
            epsilon = 1e-1
        );
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

        let viewport = Viewport {
            width: 512,
            height: 512,
        };
        let bytecode = data.iter_asm().collect::<Vec<_>>();
        // debug!("{:?}", bytecode);
        let result = pollster::block_on(evaluate_tape(&bytecode, viewport));
        assert_relative_eq!(
            result.unwrap().as_slice(),
            jit_evaluate(&tree, viewport).as_slice(),
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

    fn jit_evaluate(tree: &Tree, viewport: Viewport) -> Vec<f32> {
        let shape = JitShape::from(tree.clone());
        let tape = shape.ez_float_slice_tape();
        let mut eval = JitShape::new_float_slice_eval();

        let (x, y, z) = grid_sample(
            viewport.width as f32 - 1.0,
            viewport.height as f32 - 1.0,
            viewport.width,
            viewport.height,
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
