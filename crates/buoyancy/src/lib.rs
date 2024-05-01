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

    pub fn displacement(&self) -> MultiPolygon {
        let boat_bounding_box = self.geometry.bounding_rect().unwrap();
        let water = Rect::new(
            boat_bounding_box.min(),
            Coord {
                x: boat_bounding_box.max().x,
                y: 0.0,
            },
        )
        .to_polygon();

        water.intersection(&self.geometry)
    }

    pub fn update(
        &self,
        accelerations: Accelerations,
        linear_damping: f64,
        angular_damping: f64,
    ) -> Boat {
        let mut boat = self.clone();

        boat.geometry.rotate_around_point_mut(
            accelerations.angular_acceleration * angular_damping,
            boat.center_of_gravity(),
        );

        boat.geometry
            .translate_mut(0.0, accelerations.vertical_acceleration * linear_damping);

        boat
    }
}

pub struct Simulation {
    pub density_water: f64,
    pub gravity: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Simulation {
    pub fn new() -> Self {
        Self {
            density_water: 1., // kg / L
            gravity: 9.8,      // m / s^2
            tolerance: 1e-4,   // (kg m) / s^2
            max_iterations: 1000,
        }
    }

    pub fn compute_accelerations(&self, boat: &Boat) -> Accelerations {
        let center_of_gravity = boat.center_of_gravity();

        // Calculate net vertical force
        let displacement = boat.displacement();
        let boat_mass = boat.density * boat.geometry.unsigned_area();
        let force_gravity = -self.gravity * boat_mass;

        let (force_buoyancy, torque) = {
            match displacement.centroid() {
                None => (0.0, 0.0),
                Some(center_of_buoyancy) => {
                    let force_buoyancy =
                        self.density_water * displacement.unsigned_area() * self.gravity;
                    let distance_vector = center_of_buoyancy - center_of_gravity;
                    let torque = distance_vector.x() * force_buoyancy;
                    (force_buoyancy, torque)
                }
            }
        };

        let force_net = force_buoyancy + force_gravity;
        let vertical_acceleration = force_net / boat_mass;

        let moment_of_inertia = 1.0; // We don't need this since we don't care about making the simulation physical.
        let angular_acceleration = torque / moment_of_inertia;

        Accelerations {
            vertical_acceleration,
            angular_acceleration,
        }
    }

    pub fn step(&self, boat: &Boat) -> (Boat, bool) {
        let accelerations = self.compute_accelerations(boat);
        let boat = boat.update(accelerations, 0.01, 0.5);
        let converged = accelerations.negligible(self.tolerance);
        (boat, converged)
    }

    pub fn run(&self, boat: &Boat) -> (Boat, bool) {
        let mut iterations = 0;
        let mut boat = boat.clone();
        let mut converged = false;
        while !converged && iterations < self.max_iterations {
            (boat, converged) = self.step(&boat);
            iterations += 1;
            dbg!(iterations);
        }
        (boat, converged)
    }
}
