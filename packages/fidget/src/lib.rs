use fidget::context::{Context, Node};
use fidget::render::{BitRenderMode, RenderConfig};
use fidget::var::Var;
use fidget::vm::VmShape;
use std::ffi::CString;
use std::os::raw::c_char;

use wasm_bindgen::prelude::*;

#[no_mangle]
pub fn new_context() -> Box<Context> {
    Box::new(Context::new())
}

#[no_mangle]
pub fn ctx_x(ctx: &mut Context) -> Node {
    ctx.x()
}

#[no_mangle]
pub fn ctx_var(ctx: &mut Context) -> Node {
    ctx.var(Var::new())
}

#[no_mangle]
pub fn ctx_y(ctx: &mut Context) -> Node {
    ctx.y()
}

#[no_mangle]
pub fn ctx_z(ctx: &mut Context) -> Node {
    ctx.z()
}

#[no_mangle]
pub fn ctx_constant(ctx: &mut Context, val: f64) -> Node {
    ctx.constant(val)
}

#[no_mangle]
pub fn ctx_add(ctx: &mut Context, a: Node, b: Node) -> Node {
    ctx.add(a, b).unwrap()
}

#[no_mangle]
pub fn ctx_sub(ctx: &mut Context, a: Node, b: Node) -> Node {
    ctx.sub(a, b).unwrap()
}

#[no_mangle]
pub fn ctx_mul(ctx: &mut Context, a: Node, b: Node) -> Node {
    ctx.mul(a, b).unwrap()
}

#[no_mangle]
pub fn ctx_div(ctx: &mut Context, a: Node, b: Node) -> Node {
    ctx.div(a, b).unwrap()
}

#[no_mangle]
pub fn ctx_neg(ctx: &mut Context, a: Node) -> Node {
    ctx.neg(a).unwrap()
}

#[no_mangle]
pub fn ctx_max(ctx: &mut Context, a: Node, b: Node) -> Node {
    ctx.max(a, b).unwrap()
}

#[no_mangle]
pub fn ctx_min(ctx: &mut Context, a: Node, b: Node) -> Node {
    ctx.min(a, b).unwrap()
}

#[no_mangle]
pub fn ctx_square(ctx: &mut Context, a: Node) -> Node {
    ctx.square(a).unwrap()
}

#[no_mangle]
pub fn ctx_sqrt(ctx: &mut Context, a: Node) -> Node {
    ctx.sqrt(a).unwrap()
}

#[no_mangle]
pub fn ctx_and(ctx: &mut Context, a: Node, b: Node) -> Node {
    ctx.and(a, b).unwrap()
}

#[no_mangle]
pub fn ctx_or(ctx: &mut Context, a: Node, b: Node) -> Node {
    ctx.or(a, b).unwrap()
}

#[no_mangle]
pub fn ctx_not(ctx: &mut Context, a: Node) -> Node {
    ctx.not(a).unwrap()
}

#[no_mangle]
pub fn ctx_recip(ctx: &mut Context, a: Node) -> Node {
    ctx.recip(a).unwrap()
}

#[no_mangle]
pub fn ctx_abs(ctx: &mut Context, a: Node) -> Node {
    ctx.abs(a).unwrap()
}

#[no_mangle]
pub fn ctx_sin(ctx: &mut Context, a: Node) -> Node {
    ctx.sin(a).unwrap()
}

#[no_mangle]
pub fn ctx_cos(ctx: &mut Context, a: Node) -> Node {
    ctx.cos(a).unwrap()
}

#[no_mangle]
pub fn ctx_tan(ctx: &mut Context, a: Node) -> Node {
    ctx.tan(a).unwrap()
}

#[no_mangle]
pub fn ctx_asin(ctx: &mut Context, a: Node) -> Node {
    ctx.asin(a).unwrap()
}

#[no_mangle]
pub fn ctx_acos(ctx: &mut Context, a: Node) -> Node {
    ctx.acos(a).unwrap()
}

#[no_mangle]
pub fn ctx_atan(ctx: &mut Context, a: Node) -> Node {
    ctx.atan(a).unwrap()
}

#[no_mangle]
pub fn ctx_atan2(ctx: &mut Context, a: Node, b: Node) -> Node {
    ctx.atan2(a, b).unwrap()
}

#[no_mangle]
pub fn ctx_exp(ctx: &mut Context, a: Node) -> Node {
    ctx.exp(a).unwrap()
}

#[no_mangle]
pub fn ctx_ln(ctx: &mut Context, a: Node) -> Node {
    ctx.ln(a).unwrap()
}

#[no_mangle]
pub fn ctx_compare(ctx: &mut Context, a: Node, b: Node) -> Node {
    ctx.compare(a, b).unwrap()
}

#[no_mangle]
pub fn ctx_mod(ctx: &mut Context, a: Node, b: Node) -> Node {
    ctx.modulo(a, b).unwrap()
}

#[no_mangle]
pub fn ctx_deriv(ctx: &mut Context, node: Node, var: &Var) -> Node {
    ctx.deriv(node, *var).unwrap()
}

#[no_mangle]
pub fn ctx_to_graphviz(ctx: &mut Context) -> *mut c_char {
    let result = ctx.dot();
    CString::new(result).unwrap().into_raw()
}

#[no_mangle]
pub fn ctx_eval_node(ctx: &mut Context, node: Node) -> f64 {
    ctx.eval_xyz(node, 0., 0., 0.).unwrap()
}

#[no_mangle]
pub fn ctx_render_node(ctx: &Context, node: Node, image_size: usize) -> usize {
    let shape = VmShape::new(ctx, node).unwrap();
    let cfg = RenderConfig::<2> {
        image_size,
        ..RenderConfig::default()
    };

    let m: Vec<u8> = cfg
        .run::<_, BitRenderMode>(shape)
        .unwrap()
        .into_iter()
        .flat_map(|b| {
            let b = b as u8 * u8::MAX;
            [b, b, b, 255]
        })
        .collect();

    2
}
