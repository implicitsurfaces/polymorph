use fidget::render::{BitRenderMode, ImageRenderConfig, ImageSize, SdfRenderMode};
use fidget::vm::VmShape;

extern crate console_error_panic_hook;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Context {
    inner: fidget::context::Context,
}

#[wasm_bindgen]
pub struct Node {
    inner: fidget::context::Node,
}

impl From<fidget::context::Node> for Node {
    fn from(inner: fidget::context::Node) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen]
pub struct Var {
    inner: fidget::var::Var,
}

impl From<fidget::var::Var> for Var {
    fn from(inner: fidget::var::Var) -> Self {
        Self { inner }
    }
}

#[wasm_bindgen]
impl Context {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: fidget::context::Context::new(),
        }
    }

    #[wasm_bindgen]
    pub fn x(&mut self) -> Node {
        self.inner.x().into()
    }

    #[wasm_bindgen]
    pub fn y(&mut self) -> Node {
        self.inner.y().into()
    }

    #[wasm_bindgen]
    pub fn z(&mut self) -> Node {
        self.inner.z().into()
    }

    #[wasm_bindgen]
    pub fn var(&mut self) -> Node {
        self.inner.var(fidget::var::Var::new()).into()
    }

    #[wasm_bindgen(js_name = explicitVar)]
    pub fn explicit_var(&mut self, var: &Var) -> Node {
        self.inner.var(var.inner).into()
    }

    #[wasm_bindgen]
    pub fn constant(&mut self, val: f64) -> Node {
        self.inner.constant(val).into()
    }

    #[wasm_bindgen]
    pub fn add(&mut self, a: &Node, b: &Node) -> Node {
        self.inner.add(a.inner, b.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn sub(&mut self, a: &Node, b: &Node) -> Node {
        self.inner.sub(a.inner, b.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn mul(&mut self, a: &Node, b: &Node) -> Node {
        self.inner.mul(a.inner, b.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn div(&mut self, a: &Node, b: &Node) -> Node {
        self.inner.div(a.inner, b.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn neg(&mut self, a: &Node) -> Node {
        self.inner.neg(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn max(&mut self, a: &Node, b: &Node) -> Node {
        self.inner.max(a.inner, b.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn min(&mut self, a: &Node, b: &Node) -> Node {
        self.inner.min(a.inner, b.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn square(&mut self, a: &Node) -> Node {
        self.inner.square(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn sqrt(&mut self, a: &Node) -> Node {
        self.inner.sqrt(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn and(&mut self, a: &Node, b: &Node) -> Node {
        self.inner.and(a.inner, b.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn or(&mut self, a: &Node, b: &Node) -> Node {
        self.inner.or(a.inner, b.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn not(&mut self, a: &Node) -> Node {
        self.inner.not(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn recip(&mut self, a: &Node) -> Node {
        self.inner.recip(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn abs(&mut self, a: &Node) -> Node {
        self.inner.abs(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn sin(&mut self, a: &Node) -> Node {
        self.inner.sin(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn cos(&mut self, a: &Node) -> Node {
        self.inner.cos(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn tan(&mut self, a: &Node) -> Node {
        self.inner.tan(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn asin(&mut self, a: &Node) -> Node {
        self.inner.asin(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn acos(&mut self, a: &Node) -> Node {
        self.inner.acos(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn atan(&mut self, a: &Node) -> Node {
        self.inner.atan(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn atan2(&mut self, a: &Node, b: &Node) -> Node {
        self.inner.atan2(a.inner, b.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn exp(&mut self, a: &Node) -> Node {
        self.inner.exp(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn ln(&mut self, a: &Node) -> Node {
        self.inner.ln(a.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn compare(&mut self, a: &Node, b: &Node) -> Node {
        self.inner.compare(a.inner, b.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn modulo(&mut self, a: &Node, b: &Node) -> Node {
        self.inner.modulo(a.inner, b.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn deriv(&mut self, node: &Node, var: &Var) -> Node {
        self.inner.deriv(node.inner, var.inner).unwrap().into()
    }

    #[wasm_bindgen]
    pub fn to_graphviz(&mut self) -> String {
        self.inner.dot()
    }

    #[wasm_bindgen(js_name = evalNode)]
    pub fn eval_node(&mut self, node: &Node) -> f64 {
        self.inner.eval_xyz(node.inner, 0., 0., 0.).unwrap()
    }

    #[wasm_bindgen(js_name = evalNodeXYZ)]
    pub fn eval_node_xyz(&mut self, node: &Node, x: f64, y: f64, z: f64) -> f64 {
        self.inner.eval_xyz(node.inner, x, y, z).unwrap()
    }

    #[wasm_bindgen(js_name = renderNode)]
    pub async fn render_node(
        &self,
        node: &Node,
        image_size: usize,
        sdf_mode: Option<bool>,
    ) -> Vec<u8> {
        let shape = VmShape::new(&self.inner, node.inner).unwrap();

        let cfg = ImageRenderConfig {
            image_size: ImageSize::from(image_size as u32),
            ..ImageRenderConfig::default()
        };

        if sdf_mode.unwrap_or(false) {
            let out = cfg
                .run::<_, SdfRenderMode>(shape)
                .into_iter()
                .flat_map(|[r, g, b]| [r, g, b, 255])
                .collect();
            out
        } else {
            let out = cfg
                .run::<_, BitRenderMode>(shape)
                .into_iter()
                .map(|b| if b { 1 } else { 0 })
                .collect();
            out
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
