use buoyancy::*;
pub fn main() {
    let boat = Boat::new_default();
    dbg!(centers_of_buoyancy(&boat, 100));
}
