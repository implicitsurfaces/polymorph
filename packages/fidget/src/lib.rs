use std::sync::Once;

use fidget::render::{BitRenderMode, ImageRenderConfig, SdfPixelRenderMode, VoxelRenderConfig};
use fidget::vm::VmShape;

use gpu_interp::{GPUExpression, GPURenderConfig, Projection, Viewport};

extern crate console_error_panic_hook;
use log::*;
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

static INIT: Once = Once::new();

#[wasm_bindgen]
impl Context {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        INIT.call_once(|| {
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));

            // Initialize a logger that routes `info!`, `debug!` etc. to the console.
            console_log::init_with_level(log::Level::Debug).unwrap();
            info!("Hello from Rust!");
        });
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
        sdf_mode: bool,
        use_gpu: bool,
    ) -> Vec<u8> {
        let shape = VmShape::new(&self.inner, node.inner).unwrap();

        let cfg = ImageRenderConfig {
            image_size: (image_size as u32).into(),
            ..ImageRenderConfig::default()
        };

        if use_gpu {
            let viewport = Viewport::new(image_size as u32, image_size as u32);
            let proj = Projection::normalized_device_coords_for_viewport(viewport);
            let config = GPURenderConfig::new(viewport, proj);
            let expr = GPUExpression::new(&shape, [], config);
            let dists = gpu_interp::evaluate(&expr, None, config).await.unwrap();

            return if sdf_mode {
                let inside = [255, 255, 255, 255];
                let outside = [0, 0, 0, 255];
                dists
                    .into_iter()
                    .flat_map(|d| if d < 0.0 { inside } else { outside })
                    .collect()
            } else {
                dists
                    .into_iter()
                    .map(|d| if d < 0.0 { 1 } else { 0 })
                    .collect()
            };
        }

        if sdf_mode {
            cfg.run::<_, SdfPixelRenderMode>(shape)
                .into_iter()
                .flat_map(|[r, g, b]| [r, g, b, 255])
                .collect()
        } else {
            cfg.run::<_, BitRenderMode>(shape)
                .into_iter()
                .map(|b| if b { 1 } else { 0 })
                .collect()
        }
    }

    #[wasm_bindgen(js_name = renderNodeIn3D)]
    pub async fn render_node_in_3d(
        &self,
        node: &Node,
        image_size: usize,
        heightmap: bool,
    ) -> Vec<u8> {
        let shape = VmShape::new(&self.inner, node.inner).unwrap();

        let cfg = VoxelRenderConfig {
            image_size: (image_size as u32).into(),
            ..VoxelRenderConfig::default()
        };

        let v = cfg.run(shape);
        if heightmap {
            v.0.into_iter()
                .flat_map(|v| {
                    let d = (v as usize * 255 / image_size) as u8;
                    [d, d, d, 255]
                })
                .collect()
        } else {
            v.1.into_iter()
                .flat_map(|[r, g, b]| [r, g, b, 255])
                .collect()
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
