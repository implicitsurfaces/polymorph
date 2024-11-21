use fidget::vm::VmShape;
use gpu_interp::*;

use std::borrow::Cow;
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(&window).unwrap();
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: Some(&surface),
    };

    let (adapter, device, queue) = create_device(&instance, &options).await;

    // let viewport = {
    //     let mut size = window.inner_size();
    //     size.width = size.width.max(1);
    //     size.height = size.height.max(1);

    //     Viewport {
    //         width: size.width,
    //         height: size.height,
    //     }
    // };

    // let projection = Projection::default();

    let viewport = Viewport {
        width: 64 * 10,
        height: 16 * 4 * 10,
    };

    let projection = {
        let w = viewport.width as f32;
        let h = viewport.height as f32;
        Projection {
            scale: [1. / (w / 2.), -1. / (h / 2.)],
            translation: [1., -1.],
        }
    };

    let tape = {
        use fidget::context::Context;
        let mut file = std::fs::File::open("prospero.vm").unwrap();
        let (ctx, root) = Context::from_text(&mut file).unwrap();
        let shape = VmShape::new(&ctx, root).unwrap();

        GPUExpression::new(&shape, [], viewport.width, viewport.height)
    };

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    // Create resources shared by both pipelines.
    let (bind_group_layout, pipeline_layout) = create_pipeline_layout(
        &device,
        wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
    );
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source())),
    });
    let buffers = create_and_fill_buffers(&device, &tape, viewport, projection);
    let bind_group = create_bind_group(&device, &buffers, &bind_group_layout);

    // Create the piplines.
    let compute_pipeline = create_compute_pipeline(&device, &pipeline_layout, &shader_module);
    let render_pipeline = create_render_pipeline(
        &device,
        &pipeline_layout,
        &shader_module,
        swapchain_format,
        "fragment_main",
    );

    let mut config = surface
        .get_default_config(&adapter, viewport.width, viewport.height)
        .unwrap();
    surface.configure(&device, &config);

    let window = &window;
    let mut frame_start = Instant::now();

    let mut step_count = 0;

    event_loop
        .run(move |event, target| {
            let _ = (&instance, &adapter);

            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                match event {
                    WindowEvent::Resized(new_size) => {
                        let max_texture_size = device.limits().max_texture_dimension_2d;
                        config.width = new_size.width.max(1).min(max_texture_size);
                        config.height = new_size.height.max(1).min(max_texture_size);
                        eprint!("{} {}", config.width, config.height);
                        surface.configure(&device, &config);
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        let _frame_time = frame_start.elapsed();
                        frame_start = Instant::now();
                        // eprintln!("Frame time: {:?}", _frame_time);

                        // The step count is a "logical time" that is updated
                        // every frame.
                        step_count += 1;
                        queue.write_buffer(
                            &buffers.step_count_buffer,
                            0,
                            bytemuck::cast_slice(&[step_count]),
                        );

                        let frame = surface
                            .get_current_texture()
                            .expect("Failed to acquire next swap chain texture");
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        let timestamp_query_set =
                            device.create_query_set(&wgpu::QuerySetDescriptor {
                                label: Some("Timestamp query set"),
                                count: TIMESTAMP_COUNT as u32,
                                ty: wgpu::QueryType::Timestamp,
                            });

                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: None,
                            });

                        add_compute_pass(
                            &mut encoder,
                            &compute_pipeline,
                            &bind_group,
                            Some(&timestamp_query_set),
                            (viewport.width, viewport.height, 1),
                        );

                        add_render_pass(
                            &mut encoder,
                            &view,
                            &render_pipeline,
                            &bind_group,
                            Some(&timestamp_query_set),
                        );

                        // Resolve timestamp query results
                        encoder.resolve_query_set(
                            &timestamp_query_set,
                            0..TIMESTAMP_COUNT as u32,
                            &buffers.timestamp_resolve_buffer,
                            0,
                        );

                        // Copy timestamp results from resolve buffer to readback buffer
                        encoder.copy_buffer_to_buffer(
                            &buffers.timestamp_resolve_buffer,
                            0,
                            &buffers.timestamp_readback_buffer,
                            0,
                            TIMESTAMP_COUNT * std::mem::size_of::<u64>() as wgpu::BufferAddress,
                        );

                        queue.submit(Some(encoder.finish()));
                        frame.present();

                        pollster::block_on(print_timestamps(&device, &queue, &buffers));

                        window.request_redraw();
                    }
                    WindowEvent::CloseRequested => target.exit(),
                    _ => {}
                };
            }
        })
        .unwrap();
}

pub fn main() {
    let event_loop = EventLoop::new().unwrap();
    #[allow(unused_mut)]
    let mut builder = winit::window::WindowBuilder::new();
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowBuilderExtWebSys;
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        builder = builder.with_canvas(Some(canvas));
    }
    let window = builder.build(&event_loop).unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
