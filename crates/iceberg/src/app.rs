use buoyancy::*;
use geo::*;

use crate::ToEguiShape;

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
}

fn boat_ui_shapes(boat: &Boat, xform: AffineTransform) -> Vec<egui::Shape> {
    let mut shapes = Vec::new();
    shapes.push(
        boat.geometry
            .affine_transform(&xform)
            .to_egui_shape(egui::Color32::LIGHT_BLUE),
    );

    // Show center of gravity and buoyancy.
    for poly in boat.underwater_volume(0.0) {
        shapes.push(
            poly.affine_transform(&xform)
                .to_egui_shape(egui::Color32::YELLOW),
        );
    }

    shapes.push(
        boat.center_of_buoyancy(0.)
            .affine_transform(&xform)
            .to_egui_shape(egui::Color32::BLUE),
    );
    shapes.push(
        boat.center_of_gravity()
            .affine_transform(&xform)
            .to_egui_shape(egui::Color32::GREEN),
    );
    shapes
}

impl eframe::App for App {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        // Transform from simulation space to a reasonable spot in screen space.
        let xform =
            AffineTransform::translate(150., 150.).scaled(100., -100., Coord { x: 0., y: 0. });

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

            let water_line = geo::Line::new(
                Coord {
                    x: -10.,
                    y: water_level,
                },
                Coord {
                    x: 10.,
                    y: water_level,
                },
            )
            .affine_transform(&xform);

            ui.painter()
                .add(water_line.to_egui_shape(egui::Color32::BLUE));

            for shape in boat_ui_shapes(&self.current_boat, xform) {
                ui.painter().add(shape);
            }

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
