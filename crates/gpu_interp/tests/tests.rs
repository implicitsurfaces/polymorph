// Based on the hello_compute example from the wgpu repo.
// See https://github.com/gfx-rs/wgpu/tree/trunk/examples/src/hello_compute

use gpu_interp::*;

use approx::assert_relative_eq;
use fidget::{
    context::{Context, Tree},
    jit::JitShape,
    shape::EzShape,
    vm::VmShape,
};
use sdf::*;

#[test]
fn test_fidget_four_circles() {
    let mut circles = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            let center_x = i as f64;
            let center_y = j as f64;
            circles.push(circle(center_x, center_y, 0.5));
        }
    }
    let tree = smooth_union(circles);
    let shape = {
        let mut ctx = Context::new();
        let node = ctx.import(&tree);
        VmShape::new(&ctx, node).unwrap()
    };
    let viewport = Viewport {
        width: 256,
        height: 256,
    };

    let expr = GPUExpression::new(&shape, [], viewport.width, viewport.height);

    let result = pollster::block_on(evaluate(&expr, None, viewport));
    assert_relative_eq!(
        result.unwrap().as_slice(),
        jit_evaluate(&tree, viewport).as_slice(),
        epsilon = 1e-1
    );
}

#[test]
fn test_fidget_many_circles() {
    let mut circles = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            let center_x = i as f64;
            let center_y = j as f64;
            circles.push(circle(center_x * 200.0, center_y * 200.0, 100.0));
        }
    }
    let tree = smooth_union(circles);
    let shape = {
        let mut ctx = Context::new();
        let node = ctx.import(&tree);
        VmShape::new(&ctx, node).unwrap()
    };

    let viewport = Viewport {
        width: 256,
        height: 256,
    };
    let expr = GPUExpression::new(&shape, [], viewport.width, viewport.height);

    // debug!("{:?}", bytecode);
    let result = pollster::block_on(evaluate(&expr, None, viewport));
    assert_relative_eq!(
        result.unwrap().as_slice(),
        jit_evaluate(&tree, viewport).as_slice(),
        epsilon = 1.0
    );
}

fn circle(center_x: f64, center_y: f64, radius: f64) -> Tree {
    let dx = Tree::constant(center_x) - Tree::x();
    let dy = Tree::constant(center_y) - Tree::y();
    let dist = (dx.square() + dy.square()).sqrt();
    return dist - radius;
}

fn grid_sample(
    x_max: f32,
    y_max: f32,
    x_steps: u32,
    y_steps: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut x = Vec::with_capacity(x_steps as usize * y_steps as usize);
    let mut y = Vec::with_capacity(x_steps as usize * y_steps as usize);
    let mut z = Vec::with_capacity(x_steps as usize * y_steps as usize);

    let x_step = x_max / (x_steps - 1) as f32;
    let y_step = y_max / (y_steps - 1) as f32;

    for i in 0..y_steps {
        for j in 0..x_steps {
            let x_val = j as f32 * x_step;
            let y_val = i as f32 * y_step;

            x.push(x_val);
            y.push(y_val);
            z.push(0.0);
        }
    }

    (x, y, z)
}

fn jit_evaluate(tree: &Tree, viewport: Viewport) -> Vec<f32> {
    let shape = JitShape::from(tree.clone());
    let tape = shape.ez_float_slice_tape();
    let mut eval = JitShape::new_float_slice_eval();

    let (x, y, z) = grid_sample(
        viewport.width as f32 - 1.0,
        viewport.height as f32 - 1.0,
        viewport.width,
        viewport.height,
    );

    let start = std::time::Instant::now();
    let _ = eval.eval(&tape, x.as_slice(), y.as_slice(), z.as_slice());
    eprintln!("Jit eval took {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let r = eval.eval(&tape, x.as_slice(), y.as_slice(), z.as_slice());
    eprintln!("Jit eval #2 took {:?}", start.elapsed());
    r.unwrap().to_vec()
}
