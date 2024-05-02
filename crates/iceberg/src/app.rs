use buoyancy::*;
use egui::*;
use geo::*;

use crate::ToEguiShape;

pub struct App {
    simulation: Simulation,
    boat: Option<Boat>,
    points: Vec<Pos2>,
    convergence_error: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            points: Default::default(),
            simulation: Simulation::new(),
            boat: Some(Boat::new_default()),
            convergence_error: false,
        }
    }
}

fn to_polygon(points: &[Pos2]) -> Polygon<f64> {
    Polygon::new(
        LineString::from(
            points
                .iter()
                .map(|p| Coord {
                    x: p.x as f64,
                    y: p.y as f64,
                })
                .collect::<Vec<Coord>>(),
        ),
        vec![],
    )
    .scale_xy(1., -1.)
}

impl App {
    /// Called once before the first frame.
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        Default::default()
    }

    pub fn simulation_ui(&mut self, ui: &mut Ui, xform: &AffineTransform) {
        let boat = self.boat.as_ref().unwrap();
        let water_level = 0.;
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
        .affine_transform(xform);

        ui.painter()
            .add(water_line.to_egui_shape(egui::Color32::BLUE));

        for shape in boat_ui_shapes(boat, xform) {
            ui.painter().add(shape);
        }

        // Add a button to run the simulation
        //

        if ui.button("Run simulation").clicked() {
            let initial_boat = self.boat.clone().unwrap();
            let (boat, converged) = self.simulation.run(&initial_boat);
            self.boat = Some(boat);
            self.convergence_error = !converged;
        }

        ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
            if ui.button("Step").clicked() {
                let initial_boat = self.boat.clone().unwrap();
                let (boat, _) = self.simulation.step(&initial_boat);
                println!("Boat: {:?}", boat.center_of_gravity());
                self.boat = Some(boat);
            }

            if ui.button("⏩").clicked() {
                let mut boat = self.boat.clone().unwrap();
                for _ in 0..10 {
                    (boat, _) = self.simulation.step(&boat);
                }
                println!("Boat: {:?}", boat.center_of_gravity());
                self.boat = Some(boat);
            }
        });

        ui.with_layout(egui::Layout::left_to_right(egui::Align::TOP), |ui| {
            if ui.button("Reset").clicked() {
                self.boat = Some(Boat::new_default());
                self.convergence_error = false;
            }

            if ui.button("Draw").clicked() {
                self.boat = None;
                self.convergence_error = false;
            }
        });

        let mut density = self.boat.as_ref().unwrap().density;
        if ui
            .add(egui::Slider::new(&mut density, 0.01..=1.0).text("My value"))
            .changed()
        {
            self.boat.as_mut().unwrap().density = density;
        }

        if self.convergence_error {
            ui.label("Simulation did not converge.");
        }
    }

    pub fn boat_drawing_ui(&mut self, ui: &mut Ui) {
        ui.label("Draw a boat by clicking points on the screen.");

        let (mut response, painter) = ui.allocate_painter(
            ui.available_size_before_wrap(),
            Sense::union(Sense::click(), Sense::hover()),
        );

        let to_screen = emath::RectTransform::from_to(
            egui::Rect::from_min_size(Pos2::ZERO, response.rect.square_proportions()),
            response.rect,
        );
        let from_screen = to_screen.inverse();

        if response.clicked() {
            if let Some(pointer_pos) = response.interact_pointer_pos() {
                let canvas_pos = from_screen * pointer_pos;

                if self.points.len() > 2 && self.points.first().unwrap().distance(canvas_pos) < 0.1
                {
                    self.boat = Some(Boat {
                        geometry: to_polygon(&self.points),
                        density: 0.5,
                    });
                    self.points.clear();
                    response.mark_changed();
                } else if self.points.last() != Some(&canvas_pos) {
                    self.points.push(canvas_pos);
                    response.mark_changed();
                }
            }
        }

        let mapped_points = self
            .points
            .iter()
            .map(|p| to_screen * *p)
            .collect::<Vec<Pos2>>();

        painter.add(egui::Shape::line(
            mapped_points,
            Stroke::new(1.0, Color32::BLACK),
        ));

        if let Some(pointer_pos) = response.hover_pos() {
            if !self.points.is_empty() {
                let last_point = to_screen * *self.points.last().unwrap();

                let color = if self
                    .points
                    .first()
                    .unwrap()
                    .distance(from_screen * pointer_pos)
                    < 0.1
                {
                    Color32::GREEN
                } else {
                    Color32::BLUE
                };

                painter.add(egui::Shape::line(
                    vec![last_point, pointer_pos],
                    Stroke::new(1.0, color),
                ));
            }
        }
    }
}

/// Given a boat, return a vector of egui shapes to render.
fn boat_ui_shapes(boat: &Boat, xform: &AffineTransform) -> Vec<egui::Shape> {
    let mut shapes = Vec::new();
    shapes.push(
        boat.geometry
            .affine_transform(xform)
            .to_egui_shape(egui::Color32::LIGHT_BLUE),
    );

    // Show center of gravity and buoyancy.
    let displacement = boat.displacement();
    for poly in &displacement {
        shapes.push(
            poly.affine_transform(xform)
                .to_egui_shape(egui::Color32::YELLOW),
        );
    }

    if let Some(center_of_buoyancy) = displacement.centroid() {
        shapes.push(
            center_of_buoyancy
                .affine_transform(xform)
                .to_egui_shape(egui::Color32::BLUE),
        );
    }

    shapes.push(
        boat.center_of_gravity()
            .affine_transform(xform)
            .to_egui_shape(egui::Color32::GREEN),
    );
    shapes
}

impl eframe::App for App {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Transform from simulation space to a reasonable spot in screen space.
        let xform =
            AffineTransform::translate(150., 150.).scaled(100., -100., Coord { x: 0., y: 0. });

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
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
            if self.boat.is_some() {
                self.simulation_ui(ui, &xform)
            } else {
                self.boat_drawing_ui(ui)
            }
        });
    }
}
