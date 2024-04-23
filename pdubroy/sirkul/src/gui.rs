use egui::epaint::CubicBezierShape;
use egui::{ClippedPrimitive, Context, TexturesDelta};
use egui_wgpu::renderer::{Renderer, ScreenDescriptor};
use pixels::{wgpu, PixelsContext};
use winit::event_loop::EventLoopWindowTarget;
use winit::window::Window;

use kurbo::ParamCurveArclen;
use kurbo::{BezPath, PathEl, PathSeg, Point, Shape};

/// Manages all state required for rendering egui over `Pixels`.
pub(crate) struct Framework {
    // State for egui.
    egui_ctx: Context,
    egui_state: egui_winit::State,
    screen_descriptor: ScreenDescriptor,
    renderer: Renderer,
    paint_jobs: Vec<ClippedPrimitive>,
    textures: TexturesDelta,

    // State for the GUI
    pub gui: Gui,
}

/// Example application state. A real application will need a lot more state than this.
pub(crate) struct Gui {
    /// Only show the egui window when true.
    window_open: bool,

    pub optimizing: bool,
    pub path: BezPath,
}

impl Framework {
    /// Create egui.
    pub(crate) fn new<T>(
        event_loop: &EventLoopWindowTarget<T>,
        width: u32,
        height: u32,
        scale_factor: f32,
        pixels: &pixels::Pixels,
    ) -> Self {
        let max_texture_size = pixels.device().limits().max_texture_dimension_2d as usize;

        let egui_ctx = Context::default();
        let mut egui_state = egui_winit::State::new(event_loop);
        egui_state.set_max_texture_side(max_texture_size);
        egui_state.set_pixels_per_point(scale_factor);
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: scale_factor,
        };
        let renderer = Renderer::new(pixels.device(), pixels.render_texture_format(), None, 1);
        let textures = TexturesDelta::default();
        let gui = Gui::new();

        Self {
            egui_ctx,
            egui_state,
            screen_descriptor,
            renderer,
            paint_jobs: Vec::new(),
            textures,
            gui,
        }
    }

    /// Handle input events from the window manager.
    pub(crate) fn handle_event(&mut self, event: &winit::event::WindowEvent) {
        let _ = self.egui_state.on_event(&self.egui_ctx, event);
    }

    /// Resize egui.
    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.screen_descriptor.size_in_pixels = [width, height];
        }
    }

    /// Update scaling factor.
    pub(crate) fn scale_factor(&mut self, scale_factor: f64) {
        self.screen_descriptor.pixels_per_point = scale_factor as f32;
    }

    /// Prepare egui.
    pub(crate) fn prepare(&mut self, window: &Window) {
        // Run the egui frame and create all paint jobs to prepare for rendering.
        let raw_input = self.egui_state.take_egui_input(window);
        let output = self.egui_ctx.run(raw_input, |egui_ctx| {
            // Draw the demo application.
            self.gui.ui(egui_ctx);
        });

        self.textures.append(output.textures_delta);
        self.egui_state
            .handle_platform_output(window, &self.egui_ctx, output.platform_output);
        self.paint_jobs = self.egui_ctx.tessellate(output.shapes);
    }

    /// Render egui.
    pub(crate) fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        render_target: &wgpu::TextureView,
        context: &PixelsContext,
    ) {
        // Upload all resources to the GPU.
        for (id, image_delta) in &self.textures.set {
            self.renderer
                .update_texture(&context.device, &context.queue, *id, image_delta);
        }
        self.renderer.update_buffers(
            &context.device,
            &context.queue,
            encoder,
            &self.paint_jobs,
            &self.screen_descriptor,
        );

        // Render egui with WGPU
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: render_target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            self.renderer
                .render(&mut rpass, &self.paint_jobs, &self.screen_descriptor);
        }

        // Cleanup
        let textures = std::mem::take(&mut self.textures);
        for id in &textures.free {
            self.renderer.free_texture(id);
        }
    }
}

impl Gui {
    /// Create a `Gui`.
    fn new() -> Self {
        let mut path = BezPath::new();

        path.push(PathEl::MoveTo(Point::new(100., 200.)));
        path.push(PathEl::CurveTo(
            Point::new(100.0, 120.0),
            Point::new(120.0, 100.0),
            // Point::new(100., 200.),
            // Point::new(200., 100.),
            Point::new(200.0, 100.0),
        ));
        path.push(PathEl::CurveTo(
            Point::new(220.0, 100.0),
            Point::new(300.0, 120.0),
            // Point::new(200.0, 100.0),
            // Point::new(300.0, 200.0),
            Point::new(300.0, 200.0),
        ));
        path.push(PathEl::CurveTo(
            Point::new(300.0, 220.0),
            Point::new(220.0, 300.0),
            // Point::new(300.0, 200.0),
            // Point::new(200.0, 300.0),
            Point::new(200.0, 300.0),
        ));
        path.push(PathEl::CurveTo(
            Point::new(120.0, 300.0),
            Point::new(100.0, 220.0),
            // Point::new(200.0, 300.0),
            // Point::new(100.0, 200.0),
            Point::new(100.0, 200.0),
        ));
        path.close_path();

        let len: f64 = path
            .segments()
            .map(|seg| match seg {
                kurbo::PathSeg::Cubic(bez) => bez.arclen(0.1),
                _ => panic!("Unsupported segment type"),
            })
            .sum();
        println!("Length: {}", len);
        println!("Area: {}", path.area());
        println!("Ratio: {}", len / path.area());

        Self {
            window_open: true,
            optimizing: false,
            path,
        }
    }

    /// Create the UI using egui.
    fn ui(&mut self, ctx: &Context) {
        egui::TopBottomPanel::top("menubar_container").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("About...").clicked() {
                        self.window_open = true;
                        ui.close_menu();
                    }
                })
            });
        });
        if !self.optimizing {
            egui::Window::new("Polymorph")
                .open(&mut self.window_open)
                .show(ctx, |ui| {
                    if ui.button("Optimize!").clicked() {
                        self.optimizing = true;
                    }
                });
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            for seg in self.path.segments() {
                if let PathSeg::Cubic(b) = seg {
                    // Conver the kurbo points to egui points
                    let points =
                        [b.p0, b.p1, b.p2, b.p3].map(|p| egui::pos2(p.x as f32, p.y as f32));
                    let bezier = CubicBezierShape {
                        points,
                        closed: false,
                        fill: egui::Color32::TRANSPARENT,
                        stroke: egui::Stroke::new(2.0, egui::Color32::LIGHT_BLUE),
                    };
                    ui.painter().add(bezier);

                    // Draw lines to the control points
                    ui.painter().line_segment(
                        [points[0], points[1]],
                        egui::Stroke::new(1.0, egui::Color32::LIGHT_GRAY),
                    );
                    ui.painter().line_segment(
                        [points[2], points[3]],
                        egui::Stroke::new(1.0, egui::Color32::LIGHT_GRAY),
                    );

                    // Render handles
                    for (i, p) in points.iter().enumerate() {
                        let fill_color = if i == 0 || i == 3 {
                            egui::Color32::LIGHT_GRAY
                        } else {
                            egui::Color32::RED
                        };
                        ui.painter().rect(
                            egui::Rect::from_center_size(*p, egui::vec2(8.0, 8.0)),
                            0.0,
                            fill_color,
                            egui::Stroke::new(1.0, egui::Color32::BLACK),
                        );
                    }
                }
            }
        });
    }
}
