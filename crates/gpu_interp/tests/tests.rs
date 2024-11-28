use std::collections::HashMap;

use gpu_interp::*;

use approx::assert_relative_eq;
use fidget::{
    context::{Context, Tree},
    jit::JitShape,
    shape::EzShape,
    var::Var,
    vm::VmShape,
};
use sdf::*;

#[test]
fn test_projection() {
    let viewport = Viewport {
        width: 100,
        height: 200,
    };

    let projection = Projection::normalized_device_coords_for_viewport(viewport);
    assert!(projection.project([0., 0.]) == [50.0, 100.0]);
    assert!(projection.unproject([0., 0.]) == [-1.0, 1.0]);

    assert!(projection.unproject([50., 100.]) == [0.0, 0.0]);
    assert!(projection.project([-1.0, 1.0]) == [0.0, 0.0]);
}

#[derive(Debug)]
enum RenderError {
    #[allow(dead_code)]
    ContainsNaN(Vec<f32>),
}

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
    let viewport = Viewport::new(256, 256);
    let projection = Projection::default();
    let expr = GPUExpression::new(&shape, [], viewport, projection);
    let result = evaluate_sync(&expr, None, viewport, projection);

    assert_relative_eq!(
        result.unwrap().as_slice(),
        jit_evaluate(&tree, None, viewport).as_slice(),
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

    let viewport = Viewport::new(256, 256);
    let expr = GPUExpression::new(&shape, [], viewport, Projection::default());

    // debug!("{:?}", bytecode);
    let result = evaluate_sync(&expr, None, viewport, Default::default());
    assert_relative_eq!(
        result.unwrap().as_slice(),
        jit_evaluate(&tree, None, viewport).as_slice(),
        epsilon = 1.0
    );
}

#[test]
fn test_variable_evaluation() {
    let var_a = Var::new();
    let var_b = Var::new();

    let tree = Tree::from(var_a.clone()) + Tree::from(var_b.clone());

    let shape = {
        let mut ctx = Context::new();
        let node = ctx.import(&tree);
        VmShape::new(&ctx, node).unwrap()
    };

    let viewport = Viewport::new(256, 256);

    let bounded_vars = vec![
        BoundedVar {
            var: var_a.clone(),
            bounds: [0., 100.],
        },
        BoundedVar {
            var: var_b.clone(),
            bounds: [0., 100.],
        },
    ];

    let expr = GPUExpression::new(&shape, bounded_vars, viewport, Projection::default());

    let mut bindings = HashMap::new();
    bindings.insert(var_a, 1.0);
    bindings.insert(var_b, 2.0);

    let result = evaluate_sync(&expr, Some(&bindings), viewport, Default::default());
    assert_relative_eq!(
        result.unwrap().as_slice(),
        jit_evaluate(&tree, Some(&bindings), viewport).as_slice(),
        epsilon = 1.0
    );
}

#[test]
fn test_variable_evaluation_with_unused_var() {
    let var_a = Var::new();
    let var_b = Var::new();

    let tree = Tree::from(var_a.clone());

    let shape = {
        let mut ctx = Context::new();
        let node = ctx.import(&tree);
        VmShape::new(&ctx, node).unwrap()
    };

    let viewport = Viewport {
        width: 256,
        height: 256,
    };

    let bounded_vars = vec![
        BoundedVar {
            var: var_a.clone(),
            bounds: [0., 100.],
        },
        BoundedVar {
            var: var_b.clone(),
            bounds: [0., 100.],
        },
    ];

    let expr = GPUExpression::new(&shape, bounded_vars, viewport, Projection::default());

    let mut bindings = HashMap::new();
    bindings.insert(var_a, 1.0);
    bindings.insert(var_b, 2.0);

    let result = evaluate_sync(&expr, Some(&bindings), viewport, Default::default());
    assert_relative_eq!(
        result.unwrap().as_slice(),
        jit_evaluate(&tree, Some(&bindings), viewport).as_slice(),
        epsilon = 1.0
    );
}

#[test]
fn test_constants() {
    let tree = Tree::from(1.0);

    let shape = {
        let mut ctx = Context::new();
        let node = ctx.import(&tree);
        VmShape::new(&ctx, node).unwrap()
    };

    let viewport = Viewport::new(256, 256);
    let expr = GPUExpression::new(&shape, [], viewport, Projection::default());

    let result = evaluate_sync(&expr, None, viewport, Projection::default());
    assert_eq!(result.unwrap().as_slice(), &[1.0f32; 256 * 256]);
}

#[test]
fn test_nan_checker() {
    let tree = Tree::from(-1.0).sqrt();
    let shape = {
        let mut ctx = Context::new();
        let node = ctx.import(&tree);
        VmShape::new(&ctx, node).unwrap()
    };

    let viewport = Viewport::new(256, 256);
    let expr = GPUExpression::new(&shape, [], viewport, Projection::default());

    let result = evaluate_sync(&expr, None, viewport, Projection::default());
    assert!(result.is_err());
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

fn jit_evaluate(tree: &Tree, bindings: Option<&Bindings>, viewport: Viewport) -> Vec<f32> {
    let shape = JitShape::from(tree.clone());
    let tape = shape.ez_float_slice_tape();
    let mut eval = JitShape::new_float_slice_eval();

    let (x, y, z) = grid_sample(
        viewport.width as f32 - 1.0,
        viewport.height as f32 - 1.0,
        viewport.width,
        viewport.height,
    );

    let n = x.len();

    let start = std::time::Instant::now();
    let mut vars = HashMap::new();
    if let Some(bindings) = bindings {
        for (var, value) in bindings {
            // Only pass along vars that actually appear in the tree
            if tape.vars().get(var).is_some() {
                vars.insert(var.index().unwrap(), vec![*value; n]);
            }
        }
    }

    let _ = eval.eval_v(&tape, x.as_slice(), y.as_slice(), z.as_slice(), &vars);
    eprintln!("Jit eval took {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let r = eval.eval_v(&tape, x.as_slice(), y.as_slice(), z.as_slice(), &vars);
    eprintln!("Jit eval #2 took {:?}", start.elapsed());
    r.unwrap().to_vec()
}

fn evaluate_sync(
    expr: &GPUExpression,
    bindings: Option<&Bindings>,
    viewport: Viewport,
    projection: Projection,
) -> Result<Vec<f32>, RenderError> {
    let buf = pollster::block_on(evaluate(&expr, bindings, viewport, projection)).unwrap();
    if buf.iter().find(|v| v.is_nan()).is_some() {
        Err(RenderError::ContainsNaN(buf))
    } else {
        Ok(buf)
    }
}
