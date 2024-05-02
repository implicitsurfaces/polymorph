use buoyancy::*;
use egui::*;
use geo::*;

use crate::ToEguiShape;

pub struct App {
    simulation: Simulation,
    boat: Option<Boat>,
    boat_position: BoatPosition,
    points: Vec<Pos2>,
    convergence_error: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            points: Default::default(),
            boat_position: BoatPosition::default(),
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
    pub fn new(_: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.
        Default::default()
    }

    pub fn simulation_ui(&mut self, ctx: &Context, ui: &mut Ui, xform: &AffineTransform) {
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

        ui.painter().add(water_line.to_egui_shapes(Color32::BLUE));

        for shape in boat_ui_shapes(boat, self.boat_position, xform) {
            ui.painter().add(shape);
        }

        self.controls_ui(ctx);

        if self.convergence_error {
            ui.label("Simulation did not converge.");
        }
    }

    pub fn controls_ui(&mut self, ctx: &Context) {
        let mut draw_clicked = false;

        Window::new("Simulation")
            .anchor(Align2::RIGHT_BOTTOM, Vec2::new(-10.0, -10.0))
            .show(ctx, |ui| {
                let boat = self.boat.as_ref().unwrap();

                if ui.button("Run simulation").clicked() {
                    let (new_position, converged) = self.simulation.run(boat, self.boat_position);
                    self.boat_position = new_position;
                    self.convergence_error = !converged;
                }

                if ui.button("Orient!").clicked() {
                    match find_equilibrium_position(boat) {
                        Ok(position) => {
                            self.boat_position = position;
                            self.convergence_error = false;
                        }
                        Err(_) => {
                            self.convergence_error = true;
                        }
                    }
                }

                ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                    if ui.button("Step").clicked() {
                        let (new_position, _) = self.simulation.step(boat, self.boat_position);
                        self.boat_position = new_position;
                    }

                    if ui.button("⏩").clicked() {
                        let mut new_position = self.boat_position;
                        for _ in 0..10 {
                            (new_position, _) = self.simulation.step(boat, new_position);
                        }
                        self.boat_position = new_position;
                    }
                });

                ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                    if ui.button("Reset").clicked() {
                        self.boat = Some(Boat::new_default());
                        self.boat_position = BoatPosition::default();
                        self.convergence_error = false;
                    }

                    draw_clicked = ui.button("Draw").clicked();
                });

                ui.with_layout(Layout::left_to_right(Align::TOP), |ui| {
                    ui.label("Boat density");
                    ui.add(Slider::new(
                        &mut self.boat.as_mut().unwrap().density,
                        0.01..=1.0,
                    ))
                });
            });

        if draw_clicked {
            self.boat = None;
            self.boat_position = BoatPosition::default();
            self.convergence_error = false;
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

        painter.add(Shape::line(mapped_points, Stroke::new(1.0, Color32::BLACK)));

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

                painter.add(Shape::line(
                    vec![last_point, pointer_pos],
                    Stroke::new(1.0, color),
                ));
            }
        }
    }
}

/// Given a boat, return a vector of egui shapes to render.
fn boat_ui_shapes(boat: &Boat, position: BoatPosition, xform: &AffineTransform) -> Vec<Shape> {
    let mut shapes: Vec<Shape> = Vec::new();

    let geometry = boat.geometry_in_space(position);

    shapes.extend(
        geometry
            .affine_transform(xform)
            .to_egui_shapes(Color32::LIGHT_BLUE),
    );

    let displacement = water_displacement(boat, position);

    // Show center of gravity and buoyancy.
    for poly in &displacement {
        shapes.extend(poly.affine_transform(xform).to_egui_shapes(Color32::YELLOW));
    }

    if let Some(center_of_buoyancy) = displacement.centroid() {
        shapes.extend(
            center_of_buoyancy
                .affine_transform(xform)
                .to_egui_shapes(Color32::BLUE),
        );
    }

    shapes.extend(
        geometry
            .centroid()
            .unwrap()
            .affine_transform(xform)
            .to_egui_shapes(Color32::GREEN),
    );
    shapes
}

impl eframe::App for App {
    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        // Transform from simulation space to a reasonable spot in screen space.
        let xform =
            AffineTransform::translate(150., 150.).scaled(100., -100., Coord { x: 0., y: 0. });

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
            if self.boat.is_some() {
                self.simulation_ui(ctx, ui, &xform)
            } else {
                self.boat_drawing_ui(ui)
            }
        });
    }
}
