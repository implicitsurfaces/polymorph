use geo::polygon;
use geo::{
    Area, BooleanOps, BoundingRect, Centroid, Coord, MultiPolygon, Polygon, Rect, Rotate, Translate,
};

pub struct SimulationResults {
    dy: f64,
    angular_adjustment: f64,
}

impl SimulationResults {
    pub fn is_not_moving(&self, tolerance: f64) -> bool {
        return self.dy.abs() < tolerance && self.angular_adjustment.abs() < tolerance;
    }
}

pub struct Simulation {
    boat: Polygon<f64>,
    density_water: f64,
    density_boat: f64,
    gravity: f64,
    tolerance: f64,
    max_iterations: usize,
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

    pub fn underwater_volume(&self, boat: &Polygon<f64>, water_level: f64) -> MultiPolygon<f64> {
        let boat_bounding_box = boat.bounding_rect().unwrap();
        let water = Rect::new(
            boat_bounding_box.min(),
            Coord {
                x: boat_bounding_box.max().x,
                y: water_level,
            },
        )
        .to_polygon();

        return water.intersection(boat);
    }

    pub fn compute_forces(&self, boat: &Polygon<f64>) -> SimulationResults {
        let displacement = self.underwater_volume(boat, 0.0);

        //println!("displacement: {:?}", displacement);

        let center_of_gravity = boat.centroid().unwrap();

        let center_of_buoyancy = match displacement.centroid() {
            Some(coord) => coord,
            None => center_of_gravity,
        };

        println!("CoG: {:?}", center_of_gravity.y());
        //println!("CoB: {center_of_buoyancy:?}");

        // Calculate net vertical force
        let boat_mass = self.density_boat * boat.unsigned_area();
        let force_gravity = -self.gravity * boat_mass;
        let force_buoyancy = self.density_water * displacement.unsigned_area() * self.gravity;

        let force_net = force_buoyancy + force_gravity;

        //println!("force_net: {:?}", force_net);
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

    pub fn apply_forces(&self, boat: &mut Polygon<f64>, simulation_results: SimulationResults) {
        // 10 is a magical number so that the boat moves at a reasonable speed between steps
        let center_of_gravity = boat.centroid().unwrap();
        boat.rotate_around_point_mut(
            simulation_results.angular_adjustment / 10.,
            center_of_gravity,
        );

        boat.translate_mut(0.0, simulation_results.dy / 10.);
    }

    pub fn run(&self) -> Option<Polygon<f64>> {
        let mut converged = false;
        let mut iterations = 0;

        let mut boat = self.boat.clone();

        while !converged && iterations < self.max_iterations {
            let simulation_results = self.compute_forces(&boat);

            if simulation_results.is_not_moving(self.tolerance) {
                converged = true;
                break;
            }

            self.apply_forces(&mut boat, simulation_results);
            iterations += 1;
        }

        if !converged {
            None
        } else {
            Some(boat)
        }
    }
}
