use buoyancy::*;

fn main() {
    let simulation = Simulation::new();
    let boat = Boat::new_default();
    let (position, converged) = simulation.run(&boat);
    println!("Final position: {:?}", position);
    if !converged {
        println!("Simulation did not converge.");
    }
}
