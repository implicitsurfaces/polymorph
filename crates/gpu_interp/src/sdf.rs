use fidget::context::Tree;

pub fn circle(center_x: f64, center_y: f64, radius: f64) -> Tree {
    let dx = Tree::constant(center_x) - Tree::x();
    let dy = Tree::constant(center_y) - Tree::y();
    let dist = (dx.square() + dy.square()).sqrt();
    dist - radius
}

pub fn smooth_union(trees: Vec<Tree>) -> Tree {
    trees
        .into_iter()
        .reduce(|a, b| {
            let k = 0.1;
            let k_doubled = k * 2.0;
            let x = b.clone() - a.clone();
            0.5 * (a + b - (x.square() + k_doubled * k_doubled).sqrt())
        })
        .unwrap()
}
