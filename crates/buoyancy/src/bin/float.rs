use buoyancy::*;

fn main() {
    let simulation = Simulation::new();
    let mut boat = Boat::new_default();
    match simulation.run(&mut boat) {
        Some(results) => {
            println!("Final position: {:?}", results.center_of_gravity());
        }
        None => {
            println!("Simulation did not converge.");
        }
    }
}
