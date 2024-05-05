use buoyancy::*;
use egui::*;
use geo::*;
use iceberg::*;

pub struct App {}

impl App {
    /// Called once before the first frame.
    pub fn new(_: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.
        App {}
    }
}

impl eframe::App for App {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        let boat = Boat::new_catamaran();
        let boat_geometry = boat.geometry_in_space();

        let bb = boat_geometry.bounding_rect().unwrap();
        let window = ctx.input(|i| i.viewport().outer_rect).unwrap();
        let scale = 0.8
            * f64::min(
                window.width() as f64 / bb.width(),
                window.height() as f64 / bb.height(),
            );

        let wcx = window.width() as f64 / 2.;
        let wcy = window.height() as f64 / 2.;

        // transform to center and scale the boat
        let xform = AffineTransform::translate(
            wcx - bb.width() * scale / 2.,
            wcy + bb.height() * scale / 2.,
        )
        .scaled(scale, -scale, Coord { x: 0., y: 0. });

        TopBottomPanel::top("top_panel").show(ctx, |ui| {
            menu::bar(ui, |ui| {
                // NOTE: no File->Quit on web pages!
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }
            });
        });

        CentralPanel::default().show(ctx, |ui| {
            ui.painter().add(
                boat_geometry
                    .affine_transform(&xform)
                    .to_egui_shapes(Color32::LIGHT_BLUE),
            );

            ui.painter().add(
                boat.center_of_gravity()
                    .affine_transform(&xform)
                    .to_egui_shapes(Color32::YELLOW),
            );

            for (_angle, cob) in centers_of_buoyancy(&boat, 20) {
                ui.painter()
                    .add(cob.affine_transform(&xform).to_egui_shapes(Color32::BLUE));
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([400.0, 400.0])
            .with_min_inner_size([400.0, 400.0]),
        ..Default::default()
    };
    eframe::run_native("", native_options, Box::new(|cc| Box::new(App::new(cc))))
}
