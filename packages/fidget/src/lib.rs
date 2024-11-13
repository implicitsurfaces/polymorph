use fidget::context::{Context, Node};
use fidget::render::{BitRenderMode, RenderConfig};
use fidget::shape::Bounds;
use fidget::var::Var;
use fidget::vm::VmShape;
use std::ffi::CString;
use std::os::raw::c_char;

extern crate console_error_panic_hook;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct FidgetContext {
    ctx: Box<fidget::context::Context>,
}

#[wasm_bindgen]
pub struct FidgetNode {
    node: Box<fidget::context::Node>,
}

impl FidgetNode {
    pub fn new(node: Node) -> Self {
        let node = Box::new(node);
        Self { node }
    }
}

#[wasm_bindgen]
pub struct FidgetVar {
    var: Box<fidget::var::Var>,
}

#[wasm_bindgen]
impl FidgetVar {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let var = Box::new(Var::new());
        Self { var }
    }

    #[wasm_bindgen]
    pub fn X() -> Self {
        let var = Box::new(Var::X);
        Self { var }
    }

    #[wasm_bindgen]
    pub fn Y() -> Self {
        let var = Box::new(Var::Y);
        Self { var }
    }

    #[wasm_bindgen]
    pub fn Z() -> Self {
        let var = Box::new(Var::Z);
        Self { var }
    }
}

impl Default for FidgetVar {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl FidgetContext {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        let ctx = Box::new(Context::new());
        Self { ctx }
    }

    #[wasm_bindgen]
    pub fn x(&mut self) -> FidgetNode {
        FidgetNode::new(self.ctx.x())
    }

    #[wasm_bindgen]
    pub fn y(&mut self) -> FidgetNode {
        FidgetNode::new(self.ctx.y())
    }

    #[wasm_bindgen]
    pub fn z(&mut self) -> FidgetNode {
        FidgetNode::new(self.ctx.z())
    }

    #[wasm_bindgen]
    pub fn var(&mut self) -> FidgetNode {
        FidgetNode::new(self.ctx.var(Var::new()))
    }

    #[wasm_bindgen(js_name = explicitVar)]
    pub fn explicit_var(&mut self, var: &FidgetVar) -> FidgetNode {
        FidgetNode::new(self.ctx.var(*var.var))
    }

    #[wasm_bindgen]
    pub fn constant(&mut self, val: f64) -> FidgetNode {
        FidgetNode::new(self.ctx.constant(val))
    }

    #[wasm_bindgen]
    pub fn add(&mut self, a: &FidgetNode, b: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.add(*a.node, *b.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn sub(&mut self, a: &FidgetNode, b: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.sub(*a.node, *b.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn mul(&mut self, a: &FidgetNode, b: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.mul(*a.node, *b.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn div(&mut self, a: &FidgetNode, b: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.div(*a.node, *b.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn neg(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.neg(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn max(&mut self, a: &FidgetNode, b: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.max(*a.node, *b.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn min(&mut self, a: &FidgetNode, b: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.min(*a.node, *b.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn square(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.square(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn sqrt(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.sqrt(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn and(&mut self, a: &FidgetNode, b: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.and(*a.node, *b.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn or(&mut self, a: &FidgetNode, b: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.or(*a.node, *b.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn not(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.not(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn recip(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.recip(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn abs(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.abs(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn sin(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.sin(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn cos(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.cos(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn tan(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.tan(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn asin(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.asin(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn acos(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.acos(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn atan(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.atan(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn atan2(&mut self, a: &FidgetNode, b: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.atan2(*a.node, *b.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn exp(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.exp(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn ln(&mut self, a: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.ln(*a.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn compare(&mut self, a: &FidgetNode, b: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.compare(*a.node, *b.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn modulo(&mut self, a: &FidgetNode, b: &FidgetNode) -> FidgetNode {
        FidgetNode::new(self.ctx.modulo(*a.node, *b.node).unwrap())
    }

    #[wasm_bindgen]
    pub fn deriv(&mut self, node: &FidgetNode, var: &FidgetVar) -> FidgetNode {
        FidgetNode::new(self.ctx.deriv(*node.node, *var.var).unwrap())
    }

    #[wasm_bindgen]
    pub fn to_graphviz(&mut self) -> String {
        self.ctx.dot()
    }

    #[wasm_bindgen(js_name = evalNode)]
    pub fn eval_node(&mut self, node: &FidgetNode) -> f64 {
        self.ctx.eval_xyz(*node.node, 0., 0., 0.).unwrap()
    }

    #[wasm_bindgen(js_name = renderNode)]
    pub fn render_node(&self, node: &FidgetNode, image_size: usize) -> Vec<u8> {
        let shape = VmShape::new(&self.ctx, *node.node).unwrap();

        let cfg = RenderConfig::<2> {
            image_size,
            ..RenderConfig::default()
        };

        let out = cfg.run::<_, BitRenderMode>(shape).unwrap_throw();
        out.into_iter()
            .flat_map(|b| {
                let b = b as u8 * u8::MAX;
                [b, b, b, 255]
            })
            .collect()
    }
}

impl Default for FidgetContext {
    fn default() -> Self {
        Self::new()
    }
}

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
