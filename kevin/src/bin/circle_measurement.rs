// Compare how well our count-the-pixels measurements of a circle match up to the exact ones

use polymorph_kevin::*;

pub fn main() -> anyhow::Result<()> {
    println!(
        "{:>20} {:>20} {:>20}",
        "radius", "estimated volume", "error (%)"
    );

    for radius in [0.1, 0.2, 0.3, 0.4] {
        let circle = {
            use fidget::{context::Tree, eval::MathShape, vm::VmShape};
            let x = Tree::x();
            let y = Tree::y();
            let tree = (x.square() + y.square()).sqrt() - radius;
            VmShape::from_tree(&tree)
        };

        let b = render(circle);

        // b.print();
        let estimated_volume = b.volume();
        let actual_volume = std::f64::consts::PI * radius * radius;
        let percent_error = 100.0 * (estimated_volume - actual_volume) / actual_volume;
        println!(
            "{radius:>20.3} {:>20.3} {:>20.3}",
            estimated_volume, percent_error
        );
    }

    Ok(())
}
