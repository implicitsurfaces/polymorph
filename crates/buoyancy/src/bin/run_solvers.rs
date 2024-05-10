use buoyancy::*;
use cobyla::*;
use geo::*;

// This set of params finds the same solution as the simulation.
// scaling of step size and relative accelerations in the cost function feels arbitrary, though.
pub fn find_equilibrium_position_cobyla_1(boat: &Boat) -> Result<BoatPosition, FailStatus> {
    fn position_cost(boat: &Boat) -> f64 {
        let Accelerations {
            vertical_acceleration,
            angular_acceleration,
        } = boat.accelerations();

        vertical_acceleration.powi(2) + 100. * angular_acceleration.powi(2)
    }

    let cons: Vec<&dyn Func<()>> = vec![];

    let stop_tol = StopTols {
        ftol_abs: 1e-12,
        ..StopTols::default()
    };

    let results = minimize(
        |x: &[f64], _: &mut ()| {
            // println!("x: {:?}", x);
            position_cost(&boat.with_position(&BoatPosition {
                y_position: x[0],
                rotation_angle: x[1],
            }))
        },
        &[0.0, 0.0],
        &[
            BoatPosition::bounds_y_position(),
            BoatPosition::bounds_rotation_angle(),
        ],
        &cons,
        (),
        5000,
        RhoBeg::Set(vec![0.1, 10.]),
        Some(stop_tol),
    );

    match results {
        Ok((_, x_opt, _)) => Ok(BoatPosition {
            y_position: x_opt[0],
            rotation_angle: x_opt[1],
        }),
        Err((e, _, _)) => Err(e),
    }
}

pub fn find_equilibrium_position_cobyla_2(boat: &Boat) -> Result<BoatPosition, FailStatus> {
    fn position_cost(boat: &Boat) -> f64 {
        let displaced_water_mass = boat.displacement().unsigned_area() * DENSITY_WATER;
        (displaced_water_mass - boat.mass()).powi(2)
    }

    let cons: Vec<&dyn Func<()>> = vec![];

    let stop_tol = StopTols {
        ftol_abs: 1e-12,
        ..StopTols::default()
    };

    let results = minimize(
        |x: &[f64], _: &mut ()| {
            // println!("x: {:?}", x);
            position_cost(&boat.with_position(&BoatPosition {
                y_position: x[0],
                rotation_angle: x[1],
            }))
        },
        &[0.0, 0.0],
        &[
            BoatPosition::bounds_y_position(),
            BoatPosition::bounds_rotation_angle(),
        ],
        &cons,
        (),
        5000,
        RhoBeg::Set(vec![0.1, 10.]),
        Some(stop_tol),
    );

    match results {
        Ok((_, x_opt, _)) => Ok(BoatPosition {
            y_position: x_opt[0],
            rotation_angle: x_opt[1],
        }),
        Err((e, _, _)) => Err(e),
    }
}

fn main() {
    let boat = Boat::new_default();

    let (Boat { position, .. }, converged) = Simulation::new().run(&boat);
    assert!(converged);
    println!("Simulation position: {position:?}");

    // This acceleration-cost solver finds a solution, but parameter scaling is unjustified and were chosen to get to known solution.
    println!(
        "Solver 1 position: {:?}",
        find_equilibrium_position_cobyla_1(&boat).unwrap()
    );

    // This water-displacement cost solver finds a solution, but it's not physical because it disregards the torque
    println!(
        "Solver 2 position: {:?}",
        find_equilibrium_position_cobyla_2(&boat).unwrap()
    );
}
