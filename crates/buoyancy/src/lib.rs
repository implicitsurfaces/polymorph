use geo::*;

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

pub fn compute_accelerations(boat: &Boat, position: BoatPosition) -> Accelerations {
    const DENSITY_WATER: f64 = 1.0; // kg / L
    const GRAVITY: f64 = 9.8; // m / s^2

    let center_of_gravity = boat.center_of_gravity();

    // Calculate net vertical force

    let force_gravity = -GRAVITY * boat.mass();

    let displacement = water_displacement(boat, position);
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
    let vertical_acceleration = force_net / boat.mass();

    let moment_of_inertia = 1.0;
    let angular_acceleration = torque / moment_of_inertia;

    Accelerations {
        vertical_acceleration,
        angular_acceleration,
    }
}

#[derive(Debug, Clone)]
pub struct Boat {
    pub geometry: Polygon<f64>,
    pub density: f64,
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
        }
    }

    pub fn center_of_gravity(&self) -> Point<f64> {
        self.geometry.centroid().unwrap()
    }

    pub fn volume(&self) -> f64 {
        self.geometry.unsigned_area()
    }

    pub fn mass(&self) -> f64 {
        self.density * self.volume()
    }

    pub fn geometry_in_space(&self, position: BoatPosition) -> Polygon<f64> {
        self.geometry
            .translate(0.0, position.y_position)
            .rotate_around_centroid(position.rotation_angle)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoatPosition {
    pub rotation_angle: f64,
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

#[derive(Debug, Clone)]
pub struct BuoyancyParams {
    pub water_volume: f64,
    pub center_of_buoyancy: Option<Point<f64>>,
}

pub fn water_displacement(boat: &Boat, position: BoatPosition) -> MultiPolygon<f64> {
    let geom = boat.geometry_in_space(position);
    let boat_bounding_box = geom.bounding_rect().unwrap();

    let water = Rect::new(
        boat_bounding_box.min(),
        Coord {
            x: boat_bounding_box.max().x,
            y: 0.0,
        },
    )
    .to_polygon();

    water.intersection(&geom)
}

fn update_position(
    position: BoatPosition,
    accelerations: Accelerations,
    linear_damping: f64,
    angular_damping: f64,
) -> BoatPosition {
    let new_rotation_angle =
        position.rotation_angle + accelerations.angular_acceleration * angular_damping;
    let new_y_position = position.y_position + accelerations.vertical_acceleration * linear_damping;

    BoatPosition {
        rotation_angle: new_rotation_angle,
        y_position: new_y_position,
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

    pub fn step(&self, boat: &Boat, position: BoatPosition) -> (BoatPosition, bool) {
        let accelerations = compute_accelerations(boat, position);
        let boat = update_position(position, accelerations, 0.005, 7.);
        let converged = accelerations.negligible(self.tolerance);
        (boat, converged)
    }

    pub fn run(&self, boat: &Boat, position: BoatPosition) -> (BoatPosition, bool) {
        let mut iterations = 0;
        let mut pos = position;
        let mut converged = false;
        while !converged && iterations < self.max_iterations {
            (pos, converged) = self.step(boat, pos);
            iterations += 1;
        }
        (pos, converged)
    }
}
