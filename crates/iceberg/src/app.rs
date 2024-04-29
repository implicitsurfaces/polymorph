use buoyancy::Simulation;
use geo::{Centroid, Coord, Polygon, Scale, Translate};

pub struct App {
    simulation: Simulation,
}

impl Default for App {
    fn default() -> Self {
        Self {
            simulation: Simulation::new(),
        }
    }
}

impl App {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        Default::default()
    }

    fn draw_polygon(&self, poly: &Polygon, color: egui::Color32) -> egui::Shape {
        let points = poly
            .exterior()
            .scale_around_point(100., -100., Coord { x: 0., y: 0. })
            .translate(100., 100.)
            .points()
            .map(|p| egui::pos2(p.x() as f32, p.y() as f32))
            .collect();

        let shape = egui::Shape::convex_polygon(
            points,
            color,
            egui::Stroke::new(1.0, egui::Color32::BLACK),
        );
        shape
    }
}

impl eframe::App for App {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::menu::bar(ui, |ui| {
                // NOTE: no File->Quit on web pages!
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }

                // egui::widgets::global_dark_light_mode_buttons(ui);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Render water level
            let water_level = 0.0;
            let water_line = egui::Shape::line_segment(
                [
                    egui::pos2(0.0, water_level + 100.0),
                    egui::pos2(1000.0, water_level + 100.0),
                ],
                egui::Stroke::new(1.0, egui::Color32::BLUE),
            );
            ui.painter().add(water_line);

            let boat_shape = self.draw_polygon(&self.simulation.boat, egui::Color32::LIGHT_BLUE);
            ui.painter().add(boat_shape);

            // Show center of gravity and buoyancy.
            // let displacement = self
            //     .simulation
            //     .underwater_volume(&self.simulation.boat, 0.0);
            let displacement = self
                .simulation
                .underwater_volume(&self.simulation.boat, 0.0)
                .into_iter()
                .next()
                .unwrap();
            let displacment_shape = self.draw_polygon(&displacement, egui::Color32::GREEN);
            ui.painter().add(displacment_shape);

            let cob = displacement
                .centroid()
                .unwrap()
                .translate(100., -100.)
                .scale(100.);
            let cob_pos = egui::pos2(cob.x() as f32, cob.y() as f32);
            let cob_shape = egui::Shape::circle_filled(cob_pos, 2.0, egui::Color32::GREEN);
            ui.painter().add(cob_shape);

            let cog = self
                .simulation
                .boat
                .centroid()
                .unwrap()
                .translate(100., -100.)
                .scale(100.);

            let cog_pos = egui::pos2(cog.x() as f32, cog.y() as f32);
            let cog_shape = egui::Shape::circle_filled(cog_pos, 2.0, egui::Color32::RED);
            ui.painter().add(cog_shape);
        });
    }
}
