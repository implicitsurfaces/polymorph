use geo::polygon;
use geo::{
    Area, BooleanOps, BoundingRect, Centroid, Coord, MultiPolygon, Point, Polygon, Rect, Rotate,
    Translate,
};

pub struct SimulationResults {
    pub dy: f64,
    pub angular_adjustment: f64,
}

impl SimulationResults {
    pub fn is_not_moving(&self, tolerance: f64) -> bool {
        return self.dy.abs() < tolerance && self.angular_adjustment.abs() < tolerance;
    }
}

pub struct Boat {
    pub geometry: Polygon<f64>,
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

        Self { geometry }
    }

    pub fn underwater_volume(&self, water_level: f64) -> MultiPolygon<f64> {
        let boat_bounding_box = self.geometry.bounding_rect().unwrap();
        let water = Rect::new(
            boat_bounding_box.min(),
            Coord {
                x: boat_bounding_box.max().x,
                y: water_level,
            },
        )
        .to_polygon();
        return water.intersection(&self.geometry);
    }

    pub fn center_of_gravity(&self) -> Point<f64> {
        return self.geometry.centroid().unwrap();
    }

    pub fn center_of_buoyancy(&self, water_level: f64) -> Point<f64> {
        let displacement = self.underwater_volume(water_level);

        match displacement.centroid() {
            Some(coord) => coord,
            None => self.center_of_gravity(),
        }
    }

    pub fn apply_force(&mut self, simulation_results: SimulationResults) {
        let center_of_gravity = self.geometry.centroid().unwrap();
        self.geometry.rotate_around_point_mut(
            simulation_results.angular_adjustment / 10.,
            center_of_gravity,
        );

        self.geometry
            .translate_mut(0.0, simulation_results.dy / 10.);
    }
}

pub struct Simulation {
    pub boat: Polygon<f64>,
    pub density_water: f64,
    pub density_boat: f64,
    pub gravity: f64,
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Simulation {
    pub fn new() -> Self {
        let length: f64 = 0.5;
        let boat = polygon![
            (x: 0.0, y: 0.0),
            (x: 0.0, y: length),
            (x: length, y: length),
            (x: length, y: 0.0),
        ]
        .rotate_around_center(12.);

        Self {
            boat,
            density_water: 1000.,
            density_boat: 500.,
            gravity: 10.0,
            tolerance: 1e-5,
            max_iterations: 100,
        }
    }

    pub fn compute_forces(&self, boat: &Boat) -> SimulationResults {
        let displacement = boat.underwater_volume(0.0);

        let center_of_gravity = boat.center_of_gravity();

        let center_of_buoyancy = match displacement.centroid() {
            Some(coord) => coord,
            None => center_of_gravity,
        };

        // Calculate net vertical force
        let boat_mass = self.density_boat * boat.geometry.unsigned_area();
        let force_gravity = -self.gravity * boat_mass;
        let force_buoyancy = self.density_water * displacement.unsigned_area() * self.gravity;

        let force_net = force_buoyancy + force_gravity;

        let dy = force_net / (boat_mass * self.gravity);

        // Calculate torque
        let distance_vector = center_of_buoyancy - center_of_gravity;

        let torque = distance_vector.x() * force_buoyancy; // Simplified 2D torque about z-axis
        let moment_of_inertia = 1.0; // We don't need this since we don't care about making the simulation physical.
        let angular_adjustment = torque / moment_of_inertia;

        return SimulationResults {
            dy,
            angular_adjustment,
        };
    }

    pub fn run(&self) -> Option<Boat> {
        let mut converged = false;
        let mut iterations = 0;

        let mut boat = Boat {
            geometry: self.boat.clone(),
        };

        while !converged && iterations < self.max_iterations {
            let simulation_results = self.compute_forces(&boat);

            if simulation_results.is_not_moving(self.tolerance) {
                converged = true;
                break;
            }

            boat.apply_force(simulation_results);
            iterations += 1;
        }

        if !converged {
            None
        } else {
            Some(boat)
        }
    }
}
