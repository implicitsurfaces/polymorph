use geo::*;

mod autodiff_lib;
use crate::autodiff_lib::*;

pub type Degrees = FT<f64>;

const WATER_LEVEL: f64 = 0.0;
const DENSITY_WATER: f64 = 1.0; // kg / L
const GRAVITY: f64 = 9.8; // m / s^2

#[derive(Debug, Clone, Copy)]
pub struct Accelerations {
    pub vertical_acceleration: FT<f64>,
    pub angular_acceleration: FT<f64>,
}

impl Accelerations {
    pub fn negligible(&self, tolerance: f64) -> bool {
        self.vertical_acceleration.value().abs() < tolerance
            && self.angular_acceleration.value().abs() < tolerance
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoatPosition {
    pub rotation_angle: Degrees,
    pub y_position: FT<f64>,
}

impl Default for BoatPosition {
    fn default() -> Self {
        Self {
            rotation_angle: F::var(0.0),
            y_position: F::var(0.0),
        }
    }
}

impl BoatPosition {
    fn update(
        &self,
        accelerations: Accelerations,
        linear_damping: FT<f64>,
        angular_damping: FT<f64>,
    ) -> Self {
        BoatPosition {
            rotation_angle: self.rotation_angle
                + accelerations.angular_acceleration * angular_damping,
            y_position: self.y_position + accelerations.vertical_acceleration * linear_damping,
        }
    }

    fn bounds_rotation_angle() -> (f64, f64) {
        (-180., 180.)
    }

    fn bounds_y_position() -> (f64, f64) {
        (-100.0, 100.0)
    }
}

#[derive(Debug, Clone)]
pub struct Boat {
    pub geometry: Polygon<FT<f64>>,
    pub density: FT<f64>,
    pub position: BoatPosition,
}

impl Boat {
    pub fn new_default() -> Self {
        let length = F::cst(0.5);
        let geometry: Polygon<FT<f64>> = polygon![
            (x: 0.0.into(), y: 0.0.into()),
            (x: 0.0.into(), y: length),
            (x: length, y: length),
            (x: length, y: 0.0.into()),
        ];

        Self {
            geometry,
            density: F::cst(0.5),
            position: Default::default(),
        }
    }

    pub fn with_position(&self, position: &BoatPosition) -> Boat {
        let mut boat = self.clone();
        boat.position = *position;
        boat
    }

    pub fn geometry_in_space(&self) -> Polygon<FT<f64>> {
        let t = self
            .geometry
            .translate(F::cst(0.0), self.position.y_position);

        t.rotate_around_centroid(self.position.rotation_angle)
    }

    pub fn center_of_gravity(&self) -> Point<FT<f64>> {
        self.geometry_in_space().centroid().unwrap()
    }

    pub fn volume(&self) -> FT<f64> {
        self.geometry.unsigned_area()
    }

    pub fn mass(&self) -> FT<f64> {
        self.density * self.volume()
    }

    pub fn displacement(&self) -> MultiPolygon<FT<f64>> {
        let geom = self.geometry_in_space();
        let boat_bounding_box = geom.bounding_rect().unwrap();

        let water = Rect::new(
            boat_bounding_box.min(),
            Coord {
                x: boat_bounding_box.max().x,
                y: F::cst(WATER_LEVEL),
            },
        )
        .to_polygon();

        water.intersection(&geom)
    }

    pub fn accelerations(&self) -> Accelerations {
        let center_of_gravity = self.center_of_gravity();

        // Calculate net vertical force
        let force_gravity: FT<f64> = -self.mass() * F::cst(GRAVITY);

        let displacement = self.displacement();
        let center_of_buoyancy = displacement.centroid();
        let (force_buoyancy, torque) = {
            match center_of_buoyancy {
                None => (F::cst(0.0), F::cst(0.0)),
                Some(center_of_buoyancy) => {
                    let water_volume = displacement.unsigned_area();
                    let force_buoyancy = water_volume * F::cst(DENSITY_WATER) * F::cst(GRAVITY);
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

pub fn find_equilibrium_position(boat: &Boat) -> Result<BoatPosition, FailStatus> {
    fn position_cost(boat: &Boat) -> FT<f64> {
        // We need to be in the water
        let displacement = boat.displacement();
        let water_volume = displacement.unsigned_area();

        if water_volume.is_zero() {
            return F::cst(1e6);
        }

        let gravity_cost =
            ((water_volume * F::cst(DENSITY_WATER)).pow(2.) - boat.mass().pow(2.)).pow(2.);

        let center_of_buoyancy = displacement.centroid().unwrap();
        let distance_vector = center_of_buoyancy - boat.geometry_in_space().centroid().unwrap();
        let torque_cost = distance_vector.x().pow(2.);

        gravity_cost + torque_cost
    }

    let start_pos = [-boat.center_of_gravity().y().value(), 0.0];

    let cost = |x: &[FT<f64>]| {
        position_cost(&boat.with_position(&BoatPosition {
            y_position: x[0],
            rotation_angle: x[1],
        }))
    };

    let g = grad(cost, &start_pos);
    println!("({}, {})", g[0], g[1]); //
}

fn polygon_area(sizes: &[FT<f64>]) -> FT<f64> {
    let geometry: Polygon<FT<f64>> = polygon![
        (x: 0.0.into(), y: 0.0.into()),
        (x: 0.0.into(), y: sizes[0]),
        (x: sizes[0], y: sizes[0]),
        (x: sizes[0], y: 0.0.into()),
        (x: sizes[1], y: sizes[1])
    ];

    geometry.unsigned_area()
}

fn main() {
    find_equilibrium_position(&Boat::new_default());
}
