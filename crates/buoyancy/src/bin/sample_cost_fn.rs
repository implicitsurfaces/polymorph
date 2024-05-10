use buoyancy::*;

fn main() {
    let boat = Boat::new_default();

    let mut y_position = -0.8;

    while y_position < 0.2 {
        let mut rotation_angle = 0.0;
        while rotation_angle < 120.0 {
            let cost = position_cost(&boat.with_position(&BoatPosition {
                y_position,
                rotation_angle,
            }));
            println!("{}, {}, {}", y_position, rotation_angle, cost);
            rotation_angle += 0.2;
        }
        y_position += 0.01;
    }
}
