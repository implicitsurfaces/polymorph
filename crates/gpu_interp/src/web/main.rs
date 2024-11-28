use fidget::{context::Context, var::Var, vm::VmShape};
use gpu_interp::*;
use std::{
    cell::RefCell,
    collections::HashMap,
    rc::Rc,
    sync::{Arc, LazyLock, Mutex},
};
use wasm_bindgen::prelude::*;

static VAR_MOUSE_X: LazyLock<Var> = LazyLock::new(|| Var::new());
static VAR_MOUSE_Y: LazyLock<Var> = LazyLock::new(|| Var::new());
static VAR_TIME: LazyLock<Var> = LazyLock::new(|| Var::new());

const CANVAS_WIDTH: u32 = 256;
const CANVAS_HEIGHT: u32 = 256;

pub enum ReDraw {
    Shape(VmShape),
    Mouse(i32, i32),
    Tick,
}

use std::borrow::Cow;

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}

pub async fn setup_gpu_pipeline(
    canvas: web_sys::HtmlCanvasElement,
) -> Arc<Mutex<dyn FnMut(ReDraw)>> {
    let instance = wgpu::Instance::default();

    let surface_target = wgpu::SurfaceTarget::Canvas(canvas);
    let surface = instance.create_surface(surface_target).unwrap();

    let options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: Some(&surface),
    };

    let (adapter, device, queue) = create_device(&instance, &options).await;

    // viewport should be closest multiple of tile size
    let viewport = Viewport {
        width: (CANVAS_WIDTH / TILE_SIZE_X) * TILE_SIZE_X,
        height: (CANVAS_HEIGHT / TILE_SIZE_Y) * TILE_SIZE_Y,
    };
    let config = GPURenderConfig {
        viewport,
        projection: Projection::default(),
    };
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

    let (bind_group_layout, pipeline_layout) = create_pipeline_layout(
        &device,
        wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
    );
    let mut buffers = create_buffers(&device, config);

    let bind_group = create_bind_group(&device, &buffers, &bind_group_layout);

    let compute_pipeline = create_compute_pipeline(&device, &pipeline_layout, &shader_module);
    let render_pipeline = create_render_pipeline(
        &device,
        &pipeline_layout,
        &shader_module,
        swapchain_format,
        "fragment_main_web",
    );

    let surface_config = surface
        .get_default_config(&adapter, viewport.width, viewport.height)
        .unwrap();
    surface.configure(&device, &surface_config);

    let mut step_count = 0;
    let mut time: f32 = 0.;

    let bvars = [
        BoundedVar {
            var: *VAR_MOUSE_X,
            bounds: [0.0, viewport.width as f32],
        },
        BoundedVar {
            var: *VAR_MOUSE_Y,
            bounds: [0.0, viewport.height as f32],
        },
        BoundedVar {
            var: *VAR_TIME,
            bounds: [0.0, f32::MAX],
        },
    ];

    let mut bindings = HashMap::new();
    bindings.insert(*VAR_MOUSE_X, 10.);
    bindings.insert(*VAR_MOUSE_Y, 10.);
    bindings.insert(*VAR_TIME, 10.);

    let mut expression = None;

    let draw = move |redraw| {
        step_count += 1;
        bindings.insert(*VAR_TIME, time);

        match redraw {
            ReDraw::Shape(s) => {
                time = 0.0;
                bindings.insert(*VAR_TIME, time);

                // Update with new shape
                expression = Some(GPUExpression::new(&s, &bvars, config));
                update_buffers(
                    &queue,
                    &mut buffers,
                    expression.as_ref().unwrap(),
                    Some(&bindings),
                    config,
                );
            }

            ReDraw::Mouse(x, y) => {
                if let Some(expression) = &expression {
                    bindings.insert(*VAR_MOUSE_X, x as f32);
                    bindings.insert(*VAR_MOUSE_Y, y as f32);

                    update_var_buffers(&queue, &mut buffers, &expression, &bindings);
                }
            }

            ReDraw::Tick => {
                time += 1.0;
                if let Some(expression) = &expression {
                    update_var_buffers(&queue, &mut buffers, &expression, &bindings);
                }
            }
        }

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

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        add_compute_pass(
            &mut encoder,
            &compute_pipeline,
            &bind_group,
            None,
            (viewport.width, viewport.height, 1),
        );
        add_render_pass(&mut encoder, &view, &render_pipeline, &bind_group, None);

        queue.submit(Some(encoder.finish()));
        frame.present();
    };
    Arc::new(Mutex::new(draw))
}

fn main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(Level::Debug).unwrap();

    info!("Hello from Rust");
    let doc = web_sys::window().unwrap().document().unwrap();

    let make_demo = |doc: &web_sys::Document, initial_text: &str| {
        let demos_container = doc.get_element_by_id("demos").unwrap();
        let demo = doc
            .create_element("div")
            .unwrap()
            .dyn_into::<web_sys::HtmlElement>()
            .unwrap();
        demo.set_class_name("demo");

        let textarea = doc
            .create_element("textarea")
            .unwrap()
            .dyn_into::<web_sys::HtmlTextAreaElement>()
            .unwrap();
        textarea.set_value(initial_text);

        demo.append_child(&textarea).unwrap();

        let mut canvases = vec![];
        for _ in 0..4 {
            let canvas = doc
                .create_element("canvas")
                .unwrap()
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .unwrap();
            demo.append_child(&canvas).unwrap();
            canvases.push(canvas);
        }

        demos_container.append_child(&demo).unwrap();
        wasm_bindgen_futures::spawn_local(async move {
            let draw_fn = setup_gpu_pipeline(canvases[0].clone()).await;
            let draw_fn_dx = setup_gpu_pipeline(canvases[1].clone()).await;
            let draw_fn_dy = setup_gpu_pipeline(canvases[2].clone()).await;
            let draw_fn_dt = setup_gpu_pipeline(canvases[3].clone()).await;

            let draw_fns = vec![
                draw_fn.clone(),
                draw_fn_dx.clone(),
                draw_fn_dy.clone(),
                draw_fn_dt.clone(),
            ];

            let mut engine = fidget::rhai::Engine::new();
            engine.set_limit(50_000); //¯\_(ツ)_/¯
            let extra_bindings = vec![
                ("mouse_x", *VAR_MOUSE_X),
                ("mouse_y", *VAR_MOUSE_Y),
                ("t", *VAR_TIME),
            ];
            let mut try_parse_draw = {
                let draw_fn = draw_fn.clone();
                move |text: &str| match engine.eval(text, &extra_bindings[..]) {
                    Ok(tree) => {
                        let mut ctx = Context::new();

                        let root = ctx.import(&tree);
                        let shape = VmShape::new(&ctx, root).unwrap();
                        draw_fn.lock().unwrap()(ReDraw::Shape(shape));

                        for (v, draw_fn) in [
                            (Var::X, &draw_fn_dx),
                            (Var::Y, &draw_fn_dy),
                            (*VAR_TIME, &draw_fn_dt),
                        ] {
                            let dfdv = ctx.deriv(root, v).unwrap();
                            let shape = VmShape::new(&ctx, dfdv).unwrap();
                            draw_fn.lock().unwrap()(ReDraw::Shape(shape));
                        }
                    }
                    Err(e) => {
                        info!("Couldn't eval text: {:?}", e)
                    }
                }
            };

            try_parse_draw(&textarea.value());

            let onchange = Closure::<dyn FnMut(web_sys::Event)>::new({
                move |e: web_sys::Event| {
                    let textarea = e
                        .target()
                        .unwrap()
                        .dyn_into::<web_sys::HtmlTextAreaElement>()
                        .unwrap();
                    let text = textarea.value();
                    try_parse_draw(&text);
                }
            });

            textarea.set_onchange(Some(onchange.as_ref().unchecked_ref()));
            onchange.forget(); // Prevent closure from being dropped

            let onmousemove = {
                let draw_fns = draw_fns.clone();
                Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |e: web_sys::MouseEvent| {
                    for draw_fn in &draw_fns {
                        draw_fn.lock().unwrap()(ReDraw::Mouse(e.offset_x(), e.offset_y()));
                    }
                })
            };

            canvases[0].set_onmousemove(Some(onmousemove.as_ref().unchecked_ref()));
            onmousemove.forget(); // Prevent closure from being dropped

            let f = Rc::new(RefCell::new(None));
            let g = f.clone();

            *g.borrow_mut() = Some(Closure::new({
                move || {
                    for draw_fn in &draw_fns {
                        draw_fn.lock().unwrap()(ReDraw::Tick);
                    }
                    // Schedule ourself for another requestAnimationFrame callback.
                    request_animation_frame(f.borrow().as_ref().unwrap());
                }
            }));

            request_animation_frame(g.borrow().as_ref().unwrap());
        });
    };
    make_demo(&doc, "x + mouse_x");
    make_demo(
        &doc,
        "
let r = t % 60;
let cx = 5;
let cy = 5;
((x - cx) * (x - cx)) + ((y - cy) * (y - cy)) - (r*r)
",
    );
}
