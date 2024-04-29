// Explore GEO APIs needed to create a shape, calculate its center of gravity, drop it into water and calculate intersections, etc.
//
// |            |
// |            |
// -------------- waterline at y = 0; y from +/- 1
// |            |
// |____________| x from +/- 1

use geo::{Area, BooleanOps, HasDimensions as _, Translate};

pub fn main() {
    use geo::polygon;

    let water = polygon![
            (x: -1.0, y: 0.0),
            (x: 1.0, y: 0.0),
            (x: 1.0, y: -1.0),
            (x: -1.0, y: -1.0),
    ];

    // a square initially right above the waterline
    let length = 0.5;
    let shape = polygon![
        (x: 0.0, y: 0.0),
        (x: 0.0, y: length),
        (x: length, y: length),
        (x: length, y: 0.0),
    ];

    // square is above the water, so no intersection
    assert!(water.intersection(&shape).is_empty());

    let draft = 0.1;
    let shape = shape.translate(0., -draft);
    let displacement = water.intersection(&shape);
    assert!(!displacement.is_empty());
    assert!((length * draft) == displacement.unsigned_area());
}
