use buoyancy::*;

fn main() {
    let simulation = Simulation::new();
    let boat = Boat::new_default();
    let (new_boat, converged) = simulation.run(&boat);
    println!("Final position: {:?}", new_boat.center_of_gravity());
    if !converged {
        println!("Simulation did not converge.");
    }
}
