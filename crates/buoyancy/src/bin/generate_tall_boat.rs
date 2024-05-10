use buoyancy::*;
use cobyla::*;
use geo::*;

// geometric parameterization: points evenly spaced along y axis, mirrored across x axis
const N: usize = 4;
const X_MIN: f64 = 0.1;
const X_MAX: f64 = 1.0;

fn make_boat(xs: &[f64]) -> Boat {
    // construct polygon with anticlockwise boundary by starting points in right quadrant going up, then over to left quadrant going down
    let right = xs.iter().enumerate().map(|(idx, x)| [*x, idx as f64]);
    let left = xs.iter().enumerate().rev().map(|(idx, x)| [-x, idx as f64]);

    let polygon = Polygon::new(LineString::from_iter(right.chain(left)), vec![]);

    Boat {
        density: 0.5,
        position: Default::default(),
        geometry: polygon,
    }
}

fn cost(xs: &[f64]) -> f64 {
    // since the points are every 1 unit vertically and mirrored across x, the maximum volume is 2 * (n-1). Let's target half that.
    let target_volume = (N - 1) as f64 * X_MAX;

    let boat = make_boat(xs);
    let (boat, converged) = Simulation::new().run(&boat);

    if !converged {
        // fail
        return 1_000_000.;
    }

    let volume_error = (target_volume - boat.volume()).powi(2);
    let height_above_water = boat
        .geometry_in_space()
        .exterior_coords_iter()
        .map(|c| c.y)
        .reduce(f64::max)
        .unwrap();

    volume_error - height_above_water
}

fn main() {
    let cons: Vec<&dyn Func<()>> = vec![];

    let stop_tol = StopTols {
        ftol_abs: 1e-12,
        ..StopTols::default()
    };

    let initial: Vec<f64> = std::iter::repeat(X_MAX).take(N).collect();
    let bounds: Vec<(f64, f64)> = std::iter::repeat((X_MIN, X_MAX)).take(N).collect();
    let results = minimize(
        |xs: &[f64], _: &mut ()| cost(xs),
        &initial,
        &bounds,
        &cons,
        (),
        1000,
        RhoBeg::All(5.0),
        Some(stop_tol),
    )
    .unwrap();

    dbg!(results);
}
