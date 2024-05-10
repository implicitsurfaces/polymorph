use cobyla::{minimize, FailStatus, Func, RhoBeg, StopTols};
use geo::*;

use argmin::{
    core::{observers::ObserverMode, CostFunction, Error, Executor},
    solver::neldermead::NelderMead,
};
use argmin_observer_slog::SlogLogger;
use ndarray::{array, Array1};

pub const WATER_LEVEL: f64 = 0.0;
pub const DENSITY_WATER: f64 = 1.0; // kg / L
pub const GRAVITY: f64 = 9.8; // m / s^2
pub type Degrees = f64;

#[derive(Debug, Clone, Copy)]
pub struct Accelerations {
    pub vertical_acceleration: f64,
    pub angular_acceleration: f64,
}

impl Accelerations {
    pub fn negligible(&self, tolerance: f64) -> bool {
        self.vertical_acceleration.abs() < tolerance && self.angular_acceleration.abs() < tolerance
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoatPosition {
    pub rotation_angle: Degrees,
    pub y_position: f64,
}

impl Default for BoatPosition {
    fn default() -> Self {
        Self {
            rotation_angle: 0.0,
            y_position: 0.0,
        }
    }
}

impl BoatPosition {
    pub fn update(
        &self,
        accelerations: Accelerations,
        linear_damping: f64,
        angular_damping: f64,
    ) -> Self {
        BoatPosition {
            rotation_angle: self.rotation_angle
                + accelerations.angular_acceleration * angular_damping,
            y_position: self.y_position + accelerations.vertical_acceleration * linear_damping,
        }
    }

    pub fn bounds_rotation_angle() -> (f64, f64) {
        (-180., 180.)
    }

    pub fn bounds_y_position() -> (f64, f64) {
        (-100.0, 100.0)
    }
}

#[derive(Debug, Clone)]
pub struct Boat {
    pub geometry: Polygon<f64>,
    pub density: f64,
    pub position: BoatPosition,
}

impl Boat {
    pub fn new_default() -> Self {
        let length: f64 = 0.5;
        let geometry = polygon![
            (x: 0.0, y: 0.0),
            (x: 0.0, y: length),
            (x: length, y: length),
            (x: length, y: 0.0),
        ]
        .rotate_around_center(12.);

        Self {
            geometry,
            density: 0.5,
            position: Default::default(),
        }
    }

    // Catamaran
    // Useful for testing multiple polygons of displaced water.
    pub fn new_catamaran() -> Self {
        let length: f64 = 0.5;
        let geometry = polygon![
            (x: 0.0, y: 0.0),
            (x: 0.0, y: length),
            (x: length, y: length),
            (x: length, y: 0.0),
            (x: length / 2., y: length / 2.)
        ]
        .rotate_around_center(12.);

        Self {
            geometry,
            density: 0.5,
            position: Default::default(),
        }
    }

    pub fn with_position(&self, position: &BoatPosition) -> Boat {
        let mut boat = self.clone();
        boat.position = *position;
        boat
    }

    pub fn geometry_in_space(&self) -> Polygon<f64> {
        self.geometry
            .translate(0.0, self.position.y_position)
            .rotate_around_centroid(self.position.rotation_angle)
    }

    pub fn center_of_gravity(&self) -> Point<f64> {
        self.geometry_in_space().centroid().unwrap()
    }

    pub fn volume(&self) -> f64 {
        self.geometry.unsigned_area()
    }

    pub fn mass(&self) -> f64 {
        self.density * self.volume()
    }

    pub fn displacement(&self) -> MultiPolygon<f64> {
        let geom = self.geometry_in_space();
        let boat_bounding_box = geom.bounding_rect().unwrap();

        let water = Rect::new(
            boat_bounding_box.min(),
            Coord {
                x: boat_bounding_box.max().x,
                y: WATER_LEVEL,
            },
        )
        .to_polygon();

        water.intersection(&geom)
    }

    pub fn accelerations(&self) -> Accelerations {
        let center_of_gravity = self.center_of_gravity();

        // Calculate net vertical force
        let force_gravity = -GRAVITY * self.mass();

        let displacement = self.displacement();
        let center_of_buoyancy = displacement.centroid();
        let (force_buoyancy, torque) = {
            match center_of_buoyancy {
                None => (0.0, 0.0),
                Some(center_of_buoyancy) => {
                    let water_volume = displacement.unsigned_area();
                    let force_buoyancy = DENSITY_WATER * water_volume * GRAVITY;
                    let distance_vector = center_of_buoyancy - center_of_gravity;
                    let torque = distance_vector.x() * force_buoyancy;
                    (force_buoyancy, torque)
                }
            }
        };

        let force_net = force_buoyancy + force_gravity;
        let vertical_acceleration = force_net / self.mass();

        let moment_of_inertia = 1.0; // TODO: make simulation more physically accurate by actually calculating moment of inertia from boat geometry and axis of rotation.
        let angular_acceleration = torque / moment_of_inertia;

        Accelerations {
            vertical_acceleration,
            angular_acceleration,
        }
    }
}

pub struct Simulation {
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new()
    }
}

impl Simulation {
    pub fn new() -> Self {
        Self {
            tolerance: 1e-4, // (kg m) / s^2
            max_iterations: 1000,
        }
    }

    pub fn step(&self, boat: &Boat) -> (Boat, bool) {
        let accelerations = boat.accelerations();
        let position = boat.position.update(accelerations, 0.005, 7.);
        let converged = accelerations.negligible(self.tolerance);
        (boat.with_position(&position), converged)
    }

    pub fn run(&self, boat: &Boat) -> (Boat, bool) {
        let mut iterations = 0;
        let mut converged = false;
        let mut boat = boat.clone();
        while !converged && iterations < self.max_iterations {
            (boat, converged) = self.step(&boat);
            iterations += 1;
        }
        (boat, converged)
    }
}

pub fn centers_of_buoyancy(boat: &Boat, num_samples: usize) -> Vec<(Degrees, Point)> {
    (0..num_samples)
        .map(|i| i as f64 * 360.0 / num_samples as f64)
        // rotate boat to angle and find equilibrium y-position
        .flat_map(|angle| {
            let mut boat = boat.clone();
            boat.position.rotation_angle = angle;

            let num_iterations = 1000;
            let f = move || {
                for _ in 0..num_iterations {
                    let old = boat.position;
                    let new = boat.position.update(boat.accelerations(), 0.005, 0.0);
                    boat.position = new;
                    if f64::abs(old.y_position - new.y_position) < 1e-6 {
                        return Some(boat);
                    }
                }
                None
            };
            f()
        })
        // find center of buoyancy and transform back into original boat's reference frame
        .map(|boat| {
            let cob = boat.displacement().centroid().unwrap();
            let cob_in_geometry_frame = cob
                .rotate_around_point(-boat.position.rotation_angle, boat.center_of_gravity())
                .translate(0., -boat.position.y_position);

            (boat.position.rotation_angle, cob_in_geometry_frame)
        })
        .collect()
}

pub fn position_cost(boat: &Boat) -> f64 {
    // Feel free to play with this function - I have move stuff around, but
    // couldn't find a good way to make it work.
    //
    // I have tried using the tangent of the angle - it did not improve the
    // results. Try again, maybe you can find a better way to make it work.

    let displacement = boat.displacement();
    let water_volume = displacement.unsigned_area();

    let center_of_gravity = boat.geometry_in_space().centroid().unwrap();

    if water_volume == 0.0 || (water_volume - boat.volume()).abs() < 1e-10 {
        // This function makes the solver move the center of gravity towards
        // the surface when totally within or outside the water.
        return (1. + center_of_gravity.y().abs()).powi(2);
    }

    let gravity_cost = ((water_volume * DENSITY_WATER).powi(2) - boat.mass().powi(2)).powi(2);

    let center_of_buoyancy = displacement.centroid().unwrap();
    let distance_vector = center_of_buoyancy - center_of_gravity;
    let torque_cost = distance_vector.x().powi(2);

    gravity_cost + torque_cost
}

pub fn find_equilibrium_position_cobyla(boat: &Boat) -> Result<BoatPosition, FailStatus> {
    let cons: Vec<&dyn Func<()>> = vec![];

    let stop_tol = StopTols {
        xtol_abs: vec![1e-6, 1e-2],
        ..StopTols::default()
    };

    let results = minimize(
        |x: &[f64], _: &mut ()| {
            println!("x: {:?}", x);
            position_cost(&boat.with_position(&BoatPosition {
                y_position: x[0],
                rotation_angle: x[1],
            }))
        },
        &[-boat.center_of_gravity().y(), 0.0],
        &[
            BoatPosition::bounds_y_position(),
            BoatPosition::bounds_rotation_angle(),
        ],
        &cons,
        (),
        500,
        // These correspond to the first step sizes. Playing with this has somehow improved
        // a little bit the results.
        RhoBeg::Set(vec![0.5, 90.]),
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

pub fn find_equilibrium_position_neldermead(boat: &Boat) -> Result<BoatPosition, FailStatus> {
    struct BoatCost {
        boat: Boat,
    }

    impl CostFunction for BoatCost {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, x: &Self::Param) -> Result<Self::Output, Error> {
            let boat = self.boat.with_position(&BoatPosition {
                y_position: x[0],
                rotation_angle: x[1],
            });

            let displacement = boat.displacement();
            let water_volume = displacement.unsigned_area();

            let center_of_gravity = boat.geometry_in_space().centroid().unwrap();

            if water_volume == 0.0 || (water_volume - boat.volume()).abs() < 1e-10 {
                // This function makes the solver move the center of gravity towards
                // the surface when totally within or outside the water.
                return Ok((1. + center_of_gravity.y().abs()).powi(2));
            }

            // The equilibrium should be at a point where gravity and buoyancy are equal
            let gravity_cost = ((water_volume * DENSITY_WATER) - boat.mass()).abs();

            let center_of_buoyancy = displacement.centroid().unwrap();
            let distance_vector = center_of_gravity - center_of_buoyancy;

            // This is an attempt to have a torque that is standardized - it should always be
            // between 0 and 1
            let centers_distance = center_of_gravity.euclidean_distance(&center_of_buoyancy);
            let torque_cost = if centers_distance > 0. {
                distance_vector.x().abs() / centers_distance
            } else {
                0.
            };

            // 0.6 is a magic number that seems to work not so well
            let cost = gravity_cost + torque_cost * 0.6;

            Ok(cost)
        }
    }

    let cost = BoatCost { boat: boat.clone() };

    let y_center = -boat.center_of_gravity().y();
    let initial = vec![
        array![y_center, 0.0],
        array![y_center + 10., 90.],
        array![y_center - 10., -90.],
    ];
    let solver = NelderMead::new(initial)
        .with_sd_tolerance(1e-8)
        .expect("Failed to create solver");

    // Run solver
    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(100))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run();

    match res {
        Ok(result) => {
            let best_params = result.state.best_param.unwrap();
            println!("best params: {:?}", best_params);
            println!("y_center: {}", y_center);
            Ok(BoatPosition {
                y_position: best_params[0],
                rotation_angle: best_params[1],
            })
        }
        Err(_) => Err(FailStatus::Failure),
    }
}

pub fn find_equilibrium_position(boat: &Boat) -> Result<BoatPosition, FailStatus> {
    find_equilibrium_position_cobyla(boat)
}
