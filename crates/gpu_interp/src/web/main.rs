use fidget::context::Context;
use gpu_interp::sdf::*;
use gpu_interp::*;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct JsSystem();

#[wasm_bindgen]
impl JsSystem {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        JsSystem()
    }
}

use std::borrow::Cow;

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

async fn run(event_loop: EventLoop<()>, window: Window) {
    // Make ourselves a 'static Window we can reference from the event loop so we don't need this noise https://users.rust-lang.org/t/wgpu-winit-weird-exception/92604
    static mut WINDOW: Option<Window> = None;
    unsafe { WINDOW = Some(window) }
    let window = unsafe { WINDOW.as_ref().unwrap() };
    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(window).unwrap();
    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: Some(&surface),
    };

    let (adapter, device, queue) = create_device(&instance, &options).await;

    // let mut circles = Vec::new();
    // for i in 0..10 {
    //     for j in 0..10 {
    //         let center_x = i as f64;
    //         let center_y = j as f64;
    //         circles.push(circle(center_x * 200.0, center_y * 200.0, 100.0));
    //     }
    // }
    // let tree = smooth_union(circles);

    let tree = circle(0., 0., 80.0);
    let window_size = window.inner_size();

    // viewport should be closest multiple of tile size
    let viewport = Viewport {
        width: (window_size.width / TILE_SIZE_X) * TILE_SIZE_X,
        height: (window_size.height / TILE_SIZE_Y) * TILE_SIZE_Y,
    };

    let mut ctx = Context::new();
    let node = ctx.import(&tree);
    let tape = GPUTape::new(ctx, node, viewport.width, viewport.height);

    // let projection = {
    //     let w = viewport.width as f32;
    //     let h = viewport.height as f32;
    //     Projection {
    //         scale: [1. / (w / 2.), -1. / (h / 2.)],
    //         translation: [1., -1.],
    //     }
    // };

    let projection = Default::default();

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Render Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&format!(
            "{}\n{}",
            shader_source(),
            include_str!("fragment_main_web.wgsl")
        ))),
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
            entry_point: "fragment_main_web",
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

    let buffers = create_and_fill_buffers(&device, &tape, viewport, projection);
    let bind_group = create_bind_group(&device, &buffers, &bind_group_layout);

    let mut config = surface
        .get_default_config(&adapter, viewport.width, viewport.height)
        .unwrap();
    surface.configure(&device, &config);

    let mut step_count = 0;
    use winit::platform::web::EventLoopExtWebSys;
    event_loop.spawn(move |event, target| {
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

                    let mut encoder = device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                    add_compute_pass(
                        &mut encoder,
                        &compute_pipeline,
                        &bind_group,
                        None,
                        &viewport,
                    );

                    // dropped and unlocks the encoder when we're done
                    // with it.
                    {
                        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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

                WindowEvent::CursorMoved {
                    position: winit::dpi::PhysicalPosition { x, y },
                    ..
                } => {
                    let tree = circle(x, y, 100.0);
                    let mut ctx = Context::new();
                    let node = ctx.import(&tree);
                    let tape = GPUTape::new(ctx, node, viewport.width, viewport.height);
                    queue.write_buffer(&buffers.bytecode_buffer, 0, &tape.to_bytes());
                }

                WindowEvent::CloseRequested => target.exit(),

                e => {
                    //info!("{:?}", e)
                }
            };
        }
    })
}

#[cfg(target_arch = "wasm32")]
fn main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(Level::Debug).unwrap();

    info!("Hello from Rust");

    let event_loop = EventLoop::new().unwrap();

    #[allow(unused_mut)]
    let mut builder = winit::window::WindowBuilder::new();

    use wasm_bindgen::JsCast;
    use winit::platform::web::WindowBuilderExtWebSys;
    let canvas = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .get_element_by_id("target-canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();

    builder = builder.with_canvas(Some(canvas));
    let window = builder.build(&event_loop).unwrap();
    info!("monitor scale factor: {}", window.scale_factor());
    let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(640, 640));

    wasm_bindgen_futures::spawn_local(run(event_loop, window));
}
