use fidget::{context::Context, var::Var, vm::VmShape};
use gpu_interp::*;
use std::{
    cell::LazyCell,
    collections::HashMap,
    sync::{Arc, Mutex},
};
use wasm_bindgen::prelude::*;

const VAR_MOUSE_X: LazyCell<Var> = LazyCell::new(|| Var::new());
const VAR_MOUSE_Y: LazyCell<Var> = LazyCell::new(|| Var::new());

pub enum ReDraw {
    Shape(VmShape),
    Mouse(i32, i32),
}

use std::borrow::Cow;

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

    // TODO: get from canvas
    let width = 256;
    let height = 256;

    // viewport should be closest multiple of tile size
    let viewport = Viewport {
        width: (width / TILE_SIZE_X) * TILE_SIZE_X,
        height: (height / TILE_SIZE_Y) * TILE_SIZE_Y,
    };

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

    let (bind_group_layout, pipeline_layout) = create_pipeline_layout(
        &device,
        wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
    );
    let mut buffers = create_buffers(&device, viewport, projection);

    let bind_group = create_bind_group(&device, &buffers, &bind_group_layout);

    let compute_pipeline = create_compute_pipeline(&device, &pipeline_layout, &shader_module);
    let render_pipeline = create_render_pipeline(
        &device,
        &pipeline_layout,
        &shader_module,
        swapchain_format,
        "fragment_main_web",
    );

    let config = surface
        .get_default_config(&adapter, viewport.width, viewport.height)
        .unwrap();

    surface.configure(&device, &config);

    let mut step_count = 0;

    let bvars = [
        BoundedVar {
            var: *VAR_MOUSE_X,
            bounds: [0.0, viewport.width.into()],
        },
        BoundedVar {
            var: *VAR_MOUSE_Y,
            bounds: [0.0, viewport.height.into()],
        },
    ];

    let mut bindings = HashMap::new();
    bindings.insert(*VAR_MOUSE_X, 10.);
    bindings.insert(*VAR_MOUSE_Y, 10.);

    let draw = move |redraw| {
        step_count += 1;
        let mut expression = None;

        match redraw {
            ReDraw::Shape(s) => {
                // Update with new shape
                expression = Some(GPUExpression::new(
                    &s,
                    &bvars,
                    viewport.width,
                    viewport.height,
                ));
                update_buffers(
                    &queue,
                    &mut buffers,
                    &expression.unwrap(),
                    Some(&bindings),
                    viewport.clone(),
                );
            }

            ReDraw::Mouse(x, y) => {
                if let Some(expression) = expression {
                    bindings.insert(*VAR_MOUSE_X, x as f32);
                    bindings.insert(*VAR_MOUSE_Y, y as f32);

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

        let canvas = doc
            .create_element("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();

        demo.append_child(&textarea).unwrap();
        demo.append_child(&canvas).unwrap();
        demos_container.append_child(&demo).unwrap();
        wasm_bindgen_futures::spawn_local(async move {
            let draw_fn = setup_gpu_pipeline(canvas.clone()).await;

            let mut engine = fidget::rhai::Engine::new();
            engine.set_limit(50_000); //¯\_(ツ)_/¯
            let extra_bindings = vec![("mouse_x", *VAR_MOUSE_X), ("mouse_y", *VAR_MOUSE_Y)];
            // &extra_bindings[..]
            let mut try_parse_draw = {
                let draw_fn = draw_fn.clone();
                move |text: &str| match engine.eval(text) {
                    Ok(tree) => {
                        info!("{:?}", tree);
                        let mut ctx = Context::new();
                        let root = ctx.import(&tree);
                        let shape = VmShape::new(&ctx, root).unwrap();
                        draw_fn.lock().unwrap()(ReDraw::Shape(shape));
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

            let onmousemove =
                Closure::<dyn FnMut(web_sys::MouseEvent)>::new(move |e: web_sys::MouseEvent| {
                    draw_fn.lock().unwrap()(ReDraw::Mouse(e.offset_x(), e.offset_y()));
                });

            canvas.set_onmousemove(Some(onmousemove.as_ref().unchecked_ref()));
            onmousemove.forget(); // Prevent closure from being dropped
        });
    };

    make_demo(&doc, "x");
    make_demo(&doc, "y");
}
