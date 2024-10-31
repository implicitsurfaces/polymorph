use gpu_interp::*;

use std::borrow::Cow;
use std::time::{Duration, Instant};
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

    let viewport = {
        let mut size = window.inner_size();
        size.width = size.width.max(1);
        size.height = size.height.max(1);

        Viewport {
            width: size.width,
            height: size.height,
        }
    };

    let tape = {
        use fidget::{
            context::{Context, Tree},
            vm::VmData,
        };
        use gpu_interp::sdf::*;

        fn circle(center_x: f64, center_y: f64, radius: f64) -> Tree {
            let dx = Tree::constant(center_x) - Tree::x();
            let dy = Tree::constant(center_y) - Tree::y();
            let dist = (dx.square() + dy.square()).sqrt();
            return dist - radius;
        }

        let mut circles = Vec::new();
        for i in 0..2 {
            for j in 0..2 {
                let center_x = i as f64;
                let center_y = j as f64;
                circles.push(circle(center_x * 200.0, center_y * 200.0, 100.0));
            }
        }
        let tree = smooth_union(circles);
        let mut ctx = Context::new();
        let node = ctx.import(&tree);
        let data = VmData::<REG_COUNT>::new(&ctx, &[node]).unwrap();

        data.iter_asm().collect::<Vec<_>>()
    };

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Render Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_source())),
    });

    // Setup compute pipeline
    let (pipeline_layout, bind_group_layout) = setup_pipeline_layout(
        &device,
        wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
    );
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "compute_main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Setup render pipeline
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: "vertex_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: "fragment_main",
            compilation_options: Default::default(),
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    let buffers = create_buffers(&device, &tape, viewport);
    let bind_group = create_bind_group(&device, &buffers, &bind_group_layout);

    let mut config = surface
        .get_default_config(&adapter, viewport.width, viewport.height)
        .unwrap();
    surface.configure(&device, &config);

    let window = &window;
    let mut last_fps_update = Instant::now();
    let mut frame_count = 0;

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
                        surface.configure(&device, &config);
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        frame_count += 1;
                        let now = Instant::now();
                        if now.duration_since(last_fps_update) >= Duration::from_secs(1) {
                            println!("FPS: {}", frame_count);
                            frame_count = 0;
                            last_fps_update = now;
                        }

                        let frame = surface
                            .get_current_texture()
                            .expect("Failed to acquire next swap chain texture");
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());
                        let mut encoder =
                            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: None,
                            });

                        // Run compute pass. TODO: Share this code!
                        {
                            let invoc_size =
                                (viewport.width / FRAGMENTS_PER_INVOCATION, viewport.height);
                            let mut cpass =
                                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: None,
                                    timestamp_writes: None,
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

                        // Run render pass
                        {
                            let mut rpass =
                                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                    label: None,
                                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                        view: &view,
                                        resolve_target: None,
                                        ops: wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                            store: wgpu::StoreOp::Store,
                                        },
                                    })],
                                    depth_stencil_attachment: None,
                                    timestamp_writes: None,
                                    occlusion_query_set: None,
                                });
                            rpass.set_pipeline(&render_pipeline);
                            rpass.set_bind_group(0, &bind_group, &[]);
                            rpass.draw(0..6, 0..1);
                        }

                        queue.submit(Some(encoder.finish()));
                        frame.present();

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
