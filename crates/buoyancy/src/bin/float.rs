use buoyancy::Simulation;

fn main() {
    let simulation = Simulation::new();
    match simulation.run() {
        Some(results) => {
            println!("Final position: {:?}", results.center_of_gravity());
        }
        None => {
            println!("Simulation did not converge.");
        }
    }
}
