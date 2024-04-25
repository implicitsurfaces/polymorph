// Try to use an optimizer to find the right radius of a circle to get to a specific volume

use cobyla::{minimize, Func, RhoBeg, StopTols};
use polymorph_kevin::*;
pub fn main() -> anyhow::Result<()> {
    let actual = 0.22; // target actual radius
    println!("Attempting to guess a radius of {actual:.2}");

    println!("{:>20} {:>20}", "estimate", "relative error");

    let cost = |radius| {
        let circle = {
            use fidget::{context::Tree, eval::MathShape, vm::VmShape};
            let x = Tree::x();
            let y = Tree::y();
            let tree = (x.square() + y.square()).sqrt() - radius;
            VmShape::from_tree(&tree)
        };

        let b = render(circle);

        let estimate = b.volume();
        let relative_error = (estimate - actual) / actual;
        println!("{estimate:>20.3} {relative_error:>20.3}");
        relative_error.abs()
    };

    let initial_guess = [0.1];
    let bounds = [(0.0, 0.45)];
    // no constraints
    let constraints: Vec<&dyn Func<()>> = vec![];
    let max_evaluations = 200;
    let stop_tolerance = StopTols {
        // Stop when error stops changing by more than 0.1%
        ftol_abs: 0.001,
        ..StopTols::default()
    };
    println!(
        "{:?}",
        minimize(
            |guess: &[f64], _data: &mut ()| cost(guess[0]),
            &initial_guess,
            &bounds,
            &constraints,
            (),
            max_evaluations,
            RhoBeg::All(0.01),
            Some(stop_tolerance),
        )
        .unwrap()
    );

    Ok(())
}
