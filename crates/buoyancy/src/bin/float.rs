// Explore GEO APIs needed to create a shape, calculate its center of gravity, drop it into water and calculate intersections, etc.
//
// |            |
// |            |
// -------------- waterline at y = 0; y from +/- 1
// |            |
// |____________| x from +/- 1

use geo::{Area, BooleanOps, Centroid, Coord, HasDimensions, Polygon, Rotate, Translate};

pub fn main() {
    use geo::polygon;

    let water = polygon![
            (x: -1.0, y: 0.0),
            (x: 1.0, y: 0.0),
            (x: 1.0, y: -1.0),
            (x: -1.0, y: -1.0),
    ];

    // a square initially right above the waterline
    let length: f64 = 0.5;
    let mut boat = polygon![
        (x: 0.0, y: 0.0),
        (x: 0.0, y: length),
        (x: length, y: length),
        (x: length, y: 0.0),
    ];

    ///////////////////
    // Constants

    let density_water = 1.0;
    let density_boat = 0.5;
    let gravity = 1.0;
    let tolerance = 1e-5;

    /////////////////////////////////////////////
    // Iterate until force and torque disappear (i.e., stability)
    // Note that each iteration isn't a physically accurate time step.
    // ChatGPT wants to do that, but most of it is unnecessary for our application;  I simplified so we have a minimal simulation
    // See: https://chat.openai.com/share/e3999208-d2c1-4162-808e-e70bbe32951c

    let mut converged = false;
    let mut iterations = 0;

    while !converged && iterations < 1000 {
        let displacement = water.intersection(&boat);

        let center_of_gravity = boat.centroid().unwrap();
        let center_of_buoyancy = displacement.centroid().unwrap();

        // println!("CoG: {center_of_gravity:?}");
        // println!("CoB: {center_of_buoyancy:?}");

        // Calculate net vertical force
        let boat_mass = density_boat * boat.unsigned_area();
        let force_gravity = -gravity * boat_mass;
        let force_buoyancy = density_water * displacement.unsigned_area() * gravity;
        let force_net = force_buoyancy + force_gravity;
        let dy = force_net / (boat_mass * gravity);

        // Calculate torque
        let distance_vector = Coord {
            x: center_of_buoyancy.x() - center_of_gravity.x(),
            y: center_of_buoyancy.y() - center_of_gravity.y(),
        };
        let torque = distance_vector.x * force_buoyancy; // Simplified 2D torque about z-axis
        let moment_of_inertia = 1.0; // We don't need this since we don't care about making the simulation physical.
        let angular_adjustment = torque / moment_of_inertia;

        boat.translate_mut(0., dy);
        boat.rotate_around_point_mut(angular_adjustment, center_of_gravity);

        if force_net.abs() < tolerance && torque.abs() < tolerance {
            converged = true;
        }

        iterations += 1;
    }
    println!("Converged? {converged}");

    println!("Boat: {boat:?}");
}
