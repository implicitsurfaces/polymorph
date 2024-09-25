// Based on the hello_compute example from the wgpu repo.
// See https://github.com/gfx-rs/wgpu/tree/trunk/examples/src/hello_compute

use std::{borrow::Cow, mem::size_of_val, str::FromStr};
use wgpu::util::DeviceExt;

#[cfg_attr(test, allow(dead_code))]
async fn run() {
    let numbers = if std::env::args().len() <= 2 {
        let default = vec![4, 4, 4, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        println!("No bytecode was provided, defaulting to {default:?}");
        default
    } else {
        std::env::args()
            .skip(2)
            .map(|s| u32::from_str(&s).expect("You must pass a list of positive integers!"))
            .collect()
    };

    let result = execute_gpu(&numbers).await.unwrap();

    println!("Output: {:?}", result);
}

#[cfg_attr(test, allow(dead_code))]
async fn execute_gpu(bytecode: &[u32]) -> Option<Vec<f32>> {
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
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await
        .unwrap();

    execute_gpu_inner(&device, &queue, bytecode).await
}

async fn execute_gpu_inner(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bytecode: &[u32],
) -> Option<Vec<f32>> {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    // Gets the size in bytes of the buffer.
    let size = size_of_val(bytecode) as wgpu::BufferAddress;
    println!("Size: {}", size);

    // Instantiates buffer with data (`bytecode`), ensuring at least 64 bytes.
    let storage_buffer = {
        let min_size = 64;
        let data_size = std::mem::size_of_val(bytecode);
        let buffer_size = std::cmp::max(data_size, min_size);

        let mut contents = vec![0u8; buffer_size];
        contents[..data_size].copy_from_slice(bytemuck::cast_slice(bytecode));

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer"),
            contents: &contents,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    };

    // Create a buffer to hold the uniform data (bytecode length)
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::cast_slice(&[bytecode.len() as u32]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create an output buffer for f32 data
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (bytecode.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create a staging buffer for reading output
    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Staging Buffer"),
        size: (bytecode.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Instantiates the bind group, specifying the binding of buffers.
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
        cpass.dispatch_workgroups(bytecode.len() as u32, 1, 1);
    }
    // Copy the result from the output buffer to the staging buffer
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &output_staging_buffer,
        0,
        (bytecode.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
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
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}

#[cfg(test)]
mod tests;
