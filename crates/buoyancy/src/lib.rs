use cobyla::{minimize, FailStatus, Func, RhoBeg, StopTols};
use geo::*;

pub const WATER_LEVEL: f64 = 0.0;

#[derive(Debug, Clone, Copy)]
pub struct Accelerations {
    pub vertical_acceleration: f64,
    pub angular_acceleration: f64,
}

impl Accelerations {
    pub fn negligible(&self, tolerance: f64) -> bool {
        self.vertical_acceleration.abs() < tolerance && self.angular_acceleration.abs() < tolerance
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoatPosition {
    pub rotation_angle: f64, // Degrees
    pub y_position: f64,
}

impl Default for BoatPosition {
    fn default() -> Self {
        Self {
            rotation_angle: 0.0,
            y_position: 0.0,
        }
    }
}

impl BoatPosition {
    fn update(
        &self,
        accelerations: Accelerations,
        linear_damping: f64,
        angular_damping: f64,
    ) -> Self {
        BoatPosition {
            rotation_angle: self.rotation_angle
                + accelerations.angular_acceleration * angular_damping,
            y_position: self.y_position + accelerations.vertical_acceleration * linear_damping,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Boat {
    pub geometry: Polygon<f64>,
    pub density: f64,
    pub position: BoatPosition,
}

impl Boat {
    pub fn new_default() -> Self {
        let length: f64 = 0.5;
        let geometry = polygon![
            (x: 0.0, y: 0.0),
            (x: 0.0, y: length),
            (x: length, y: length),
            (x: length, y: 0.0),
        ]
        .rotate_around_center(12.);

        Self {
            geometry,
            density: 0.5,
            position: Default::default(),
        }
    }

    // Catamaran
    // Useful for testing multiple polygons of displaced water.
    pub fn new_catamaran() -> Self {
        let length: f64 = 0.5;
        let geometry = polygon![
            (x: 0.0, y: 0.0),
            (x: 0.0, y: length),
            (x: length, y: length),
            (x: length, y: 0.0),
            (x: length / 2., y: length / 2.)
        ]
        .rotate_around_center(12.);

        Self {
            geometry,
            density: 0.5,
            position: Default::default(),
        }
    }

    pub fn with_position(&self, position: &BoatPosition) -> Boat {
        let mut boat = self.clone();
        boat.position = *position;
        boat
    }

    pub fn geometry_in_space(&self) -> Polygon<f64> {
        self.geometry
            .translate(0.0, self.position.y_position)
            .rotate_around_centroid(self.position.rotation_angle)
    }

    pub fn center_of_gravity(&self) -> Point<f64> {
        self.geometry_in_space().centroid().unwrap()
    }

    pub fn volume(&self) -> f64 {
        self.geometry.unsigned_area()
    }

    pub fn mass(&self) -> f64 {
        self.density * self.volume()
    }

    pub fn displacement(&self) -> MultiPolygon<f64> {
        let geom = self.geometry_in_space();
        let boat_bounding_box = geom.bounding_rect().unwrap();

        let water = Rect::new(
            boat_bounding_box.min(),
            Coord {
                x: boat_bounding_box.max().x,
                y: WATER_LEVEL,
            },
        )
        .to_polygon();

        water.intersection(&geom)
    }

    pub fn accelerations(&self) -> Accelerations {
        const DENSITY_WATER: f64 = 1.0; // kg / L
        const GRAVITY: f64 = 9.8; // m / s^2

        let center_of_gravity = self.center_of_gravity();

        // Calculate net vertical force
        let force_gravity = -GRAVITY * self.mass();

        let displacement = self.displacement();
        let center_of_buoyancy = displacement.centroid();
        let (force_buoyancy, torque) = {
            match center_of_buoyancy {
                None => (0.0, 0.0),
                Some(center_of_buoyancy) => {
                    let water_volume = displacement.unsigned_area();
                    let force_buoyancy = DENSITY_WATER * water_volume * GRAVITY;
                    let distance_vector = center_of_buoyancy - center_of_gravity;
                    let torque = distance_vector.x() * force_buoyancy;
                    (force_buoyancy, torque)
                }
            }
        };

        let force_net = force_buoyancy + force_gravity;
        let vertical_acceleration = force_net / self.mass();

        let moment_of_inertia = 1.0; // TODO: make simulation more physically accurate by actually calculating moment of inertia from boat geometry and axis of rotation.
        let angular_acceleration = torque / moment_of_inertia;

        Accelerations {
            vertical_acceleration,
            angular_acceleration,
        }
    }
}

pub struct Simulation {
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}

impl Simulation {
    pub fn new() -> Self {
        Self {
            tolerance: 1e-4, // (kg m) / s^2
            max_iterations: 1000,
        }
    }

    pub fn step(&self, boat: &Boat) -> (Boat, bool) {
        let accelerations = boat.accelerations();
        let position = boat.position.update(accelerations, 0.005, 7.);
        let converged = accelerations.negligible(self.tolerance);
        (boat.with_position(&position), converged)
    }

    pub fn run(&self, boat: &Boat) -> (Boat, bool) {
        let mut iterations = 0;
        let mut converged = false;
        let mut boat = boat.clone();
        while !converged && iterations < self.max_iterations {
            (boat, converged) = self.step(&boat);
            iterations += 1;
        }
        (boat, converged)
    }
}

fn position_cost(boat: &Boat) -> f64 {
    let density_water = 1.;

    // We need to be in the water
    let displacement = boat.displacement();
    let water_volume = displacement.unsigned_area();

    if water_volume == 0.0 {
        return 1e6;
    }

    let gravity_cost = ((water_volume * density_water).powi(2) - boat.mass().powi(2)).abs();

    let center_of_buoyancy = displacement.centroid().unwrap();
    let distance_vector = center_of_buoyancy - boat.geometry_in_space().centroid().unwrap();
    let torque_cost = distance_vector.x().powi(2);

    gravity_cost + torque_cost
}

pub fn find_equilibrium_position(boat: &Boat) -> Result<BoatPosition, FailStatus> {
    let bounds = vec![(-180., 180.), (-100.0, 100.0)];

    let cons: Vec<&dyn Func<()>> = vec![];

    let stop_tol = StopTols {
        ftol_rel: 1e-6,
        ..StopTols::default()
    };

    let results = minimize(
        |x: &[f64], _: &mut ()| {
            position_cost(&boat.with_position(&BoatPosition {
                y_position: x[0],
                rotation_angle: x[1],
            }))
        },
        &[-boat.center_of_gravity().y(), 0.0],
        &bounds,
        &cons,
        (),
        200,
        RhoBeg::All(0.5),
        Some(stop_tol),
    );

    match results {
        Ok((_, x_opt, _)) => Ok(BoatPosition {
            y_position: x_opt[0],
            rotation_angle: x_opt[1],
        }),
        Err((e, _, _)) => Err(e),
    }
}
