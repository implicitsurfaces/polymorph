use gpu_interp::*;

use std::borrow::Cow;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    let shader_base = shader_source();
    let fragment_shader = "
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
";

    // You can specify a backend (Vulkan, Metal, etc.)here if desired.
    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(&window).unwrap();
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        // Request an adapter which can render to our surface
        compatible_surface: Some(&surface),
    };

    let (adapter, device, queue) = create_device(&instance, &options).await;

    let shader = shader_base + fragment_shader;

    let viewport = {
        let mut size = window.inner_size();
        size.width = size.width.max(1);
        size.height = size.height.max(1);

        Viewport {
            width: size.width,
            height: size.height,
        }
    };

    let invoc_size = (viewport.width / FRAGMENTS_PER_INVOCATION, viewport.height);

    let tape = {
        use fidget::{
            compiler::RegOp,
            context::{Context, Tree},
            jit::JitShape,
            shape::EzShape,
            var::Var,
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
                circles.push(circle(center_x, center_y, 0.5));
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

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
    });

    let pipeline_layout = setup_pipeline_layout(&device, wgpu::ShaderStages::FRAGMENT);
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            compilation_options: Default::default(),
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,

        // Disable multisampling so that we see only exact pixel values from our shader.
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    });

    let (
        storage_buffer,
        uniform_buffer,
        dimensions_buffer,
        output_buffer,
        output_staging_buffer,
        timestamp_resolve_buffer,
        timestamp_readback_buffer,
        bind_group,
    ) = setup_buffers(
        &device,
        Pipeline::Render(&render_pipeline),
        &tape,
        invoc_size,
        viewport,
    );

    let mut config = surface
        .get_default_config(&adapter, viewport.width, viewport.height)
        .unwrap();
    surface.configure(&device, &config);

    let window = &window;
    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = (&instance, &adapter, &shader, &pipeline_layout);

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
                        // On macos the window needs to be redrawn manually after resizing
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
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
