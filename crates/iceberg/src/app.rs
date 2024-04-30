use buoyancy::{Boat, Simulation};
use geo::{Coord, Point, Polygon, Scale, Translate};

pub struct App {
    simulation: Simulation,
    current_boat: Boat,
}

impl Default for App {
    fn default() -> Self {
        Self {
            simulation: Simulation::new(),
            current_boat: Boat::new_catamaran(),
        }
    }
}

impl App {
    /// Called once before the first frame.
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        Default::default()
    }

    fn draw_polygon(&self, poly: &Polygon, color: egui::Color32) -> egui::Shape {
        let points = poly
            .exterior()
            .scale_around_point(100., -100., Coord { x: 0., y: 0. })
            .translate(150., 150.)
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

    fn draw_point(&self, in_point: &Point, color: egui::Color32) -> egui::Shape {
        let point = in_point
            .scale_around_point(100., -100., Coord { x: 0., y: 0. })
            .translate(150., 150.);

        let point = egui::pos2(point.x() as f32, point.y() as f32);
        let shape = egui::Shape::circle_filled(point, 2.0, color);
        shape
    }

    fn draw_boat(&self, ui: &egui::Ui, boat: &Boat) {
        let boat_shape = self.draw_polygon(&boat.geometry, egui::Color32::LIGHT_BLUE);
        ui.painter().add(boat_shape);

        // Show center of gravity and buoyancy.
        for poly in boat.underwater_volume(0.0) {
            let displacement_shape = self.draw_polygon(&poly, egui::Color32::YELLOW);
            ui.painter().add(displacement_shape);
        }

        ui.painter()
            .add(self.draw_point(&boat.center_of_buoyancy(0.), egui::Color32::BLUE));
        ui.painter()
            .add(self.draw_point(&boat.center_of_gravity(), egui::Color32::GREEN));
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
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Render water level
            let water_level = 0.0;
            let water_line = egui::Shape::line_segment(
                [
                    egui::pos2(0.0, water_level + 150.0),
                    egui::pos2(1000.0, water_level + 150.0),
                ],
                egui::Stroke::new(1.0, egui::Color32::BLUE),
            );
            ui.painter().add(water_line);

            self.draw_boat(ui, &self.current_boat);

            // Add a button to run the simulation
            //

            if ui.button("Run simulation").clicked() {
                match self.simulation.run(&self.current_boat) {
                    Some(results) => self.current_boat = results,
                    None => {
                        println!("Simulation did not converge.");
                    }
                }
                println!("Final shape {:?}", self.current_boat.geometry);
            }

            if ui.button("Reset").clicked() {
                self.current_boat = Boat::new_catamaran();
            }
        });
    }
}
