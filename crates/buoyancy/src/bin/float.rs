use buoyancy::Simulation;
use geo::{Centroid, Polygon};

fn main() {
    let simulation = Simulation::new();
    match simulation.run() {
        Some(results) => {
            println!("Final position: {:?}", results.centroid().unwrap());
        }
        None => {
            println!("Simulation did not converge.");
        }
    }
}
