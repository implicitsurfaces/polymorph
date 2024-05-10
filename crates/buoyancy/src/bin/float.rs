use buoyancy::*;

fn main() {
    let simulation = Simulation::new();
    let (boat, converged) = simulation.run(&Boat::new_default());
    println!("Final position: {:?}", boat.position);
    if !converged {
        println!("Simulation did not converge.");
    }
}
