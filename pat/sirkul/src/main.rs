#![deny(clippy::all)]
#![forbid(unsafe_code)]

use crate::gui::Framework;
use error_iter::ErrorIter as _;
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use rand::prelude::*;
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

use kurbo::Shape;

use cobyla::{minimize, Func, RhoBeg, StopTols};
use std::f64::consts::PI;

mod gui;

const WIDTH: u32 = 642;
const HEIGHT: u32 = 642;

const CELL_SIZE_PX: usize = WIDTH as usize / BOARD_SIZE;
const BOARD_SIZE: usize = 3;

struct World {
    cells: [bool; BOARD_SIZE * BOARD_SIZE],
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Hello Pixels + egui")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let (mut pixels, mut framework) = {
        let window_size = window.inner_size();
        let scale_factor = window.scale_factor() as f32;
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        let pixels = Pixels::new(WIDTH, HEIGHT, surface_texture)?;
        let framework = Framework::new(
            &event_loop,
            window_size.width,
            window_size.height,
            scale_factor,
            &pixels,
        );

        (pixels, framework)
    };
    let world = World::random(5);
    world.evaluate_loss();

    event_loop.run(move |event, _, control_flow| {
        // Handle input events
        if input.update(&event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape) || input.close_requested() {
                *control_flow = ControlFlow::Exit;
                return;
            }

            // Update the scale factor
            if let Some(scale_factor) = input.scale_factor() {
                framework.scale_factor(scale_factor);
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                if let Err(err) = pixels.resize_surface(size.width, size.height) {
                    log_error("pixels.resize_surface", err);
                    *control_flow = ControlFlow::Exit;
                    return;
                }
                framework.resize(size.width, size.height);
            }

            // Update internal state and request a redraw
            if framework.gui.optimizing {
                let mut problem = Problem {
                    path: &mut framework.gui.path,
                };
                problem.optimize();
                framework.gui.optimizing = false;
            }
            window.request_redraw();
        }

        match event {
            Event::WindowEvent { event, .. } => {
                // Update egui inputs
                framework.handle_event(&event);
            }
            // Draw the current frame
            Event::RedrawRequested(_) => {
                // Draw the world
                world.draw(pixels.frame_mut());

                // Prepare egui
                framework.prepare(&window);

                // Render everything together
                let render_result = pixels.render_with(|encoder, render_target, context| {
                    // Render the world texture
                    context.scaling_renderer.render(encoder, render_target);

                    // Render egui
                    framework.render(encoder, render_target, context);

                    Ok(())
                });

                // Basic error handling
                if let Err(err) = render_result {
                    log_error("pixels.render", err);
                    *control_flow = ControlFlow::Exit;
                }
            }
            _ => (),
        }
    });
}

fn log_error<E: std::error::Error + 'static>(method_name: &str, err: E) {
    error!("{method_name}() failed: {err}");
    for source in err.sources().skip(1) {
        error!("  Caused by: {source}");
    }
}

impl World {
    /// Create a new `World` instance that can draw a moving box.
    fn random(occupied_count: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut state = [
            vec![true; occupied_count],
            vec![false; BOARD_SIZE * BOARD_SIZE - occupied_count],
        ]
        .concat();
        state.shuffle(&mut rng);
        Self {
            cells: state.try_into().unwrap(),
        }
    }

    fn evaluate_loss(&self) -> i32 {
        let mut loss = 0;
        let mut occupied_count = 0;
        for (i, cell) in self.cells.iter().enumerate() {
            let x = i % BOARD_SIZE;
            let y = i / BOARD_SIZE;
            if *cell {
                occupied_count += 1;
                let is_boundary = x == 0
                    || x == BOARD_SIZE - 1
                    || y == 0
                    || y == BOARD_SIZE - 1
                    || !self.cells[(y - 1) * BOARD_SIZE + x]
                    || !self.cells[(y + 1) * BOARD_SIZE + x]
                    || !self.cells[y * BOARD_SIZE + x - 1]
                    || !self.cells[y * BOARD_SIZE + x + 1];
                // println!("{} {} {}", x, y, is_boundary);
                if is_boundary {
                    loss += 1;
                }
            }
        }
        println!("occupied_count: {}", occupied_count);
        loss
    }

    /// Update the `World` internal state.
    #[allow(dead_code)]
    fn update(&mut self) {
        let new_world = World::random(5);
        if new_world.evaluate_loss() < self.evaluate_loss() {
            self.cells = new_world.cells;
        }
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    fn draw(&self, frame: &mut [u8]) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let px_x = i % WIDTH as usize;
            let px_y = i / WIDTH as usize;

            let cell_x = px_x / CELL_SIZE_PX;
            let cell_y = px_y / CELL_SIZE_PX;

            let cell_idx = cell_y * BOARD_SIZE + cell_x;

            let rgba = if self.cells[cell_idx] {
                // white
                [0xff, 0xff, 0xff, 0xff]
            } else {
                // black
                [0x00, 0x00, 0x00, 0xff]
            };

            pixel.copy_from_slice(&rgba);
        }
    }
}

struct Problem<'a> {
    path: &'a mut kurbo::BezPath,
}

impl<'a> Problem<'a> {
    fn cost(&self, x: &[f64], _data: &mut ()) -> f64 {
        // Construct a new BezPath, with the elements of x as the control points.
        let mut path = self.path.clone();

        copy_control_points(x, &mut path);

        let len = path.perimeter(0.1);
        println!("len: {}", len);
        println!("area: {}", path.area());
        let r = len / (2. * PI);
        let diff = (path.area() - PI * r * r).abs();
        dbg!(diff * diff)
    }

    fn optimize(&mut self) {
        // Initial guess
        //        let xinit = vec![1., 1.];
        let xinit: Vec<f64> = self
            .path
            .elements()
            .iter()
            .filter_map(|el| match el {
                kurbo::PathEl::MoveTo(_) => None,
                kurbo::PathEl::CurveTo(p1, p2, p3) => {
                    Some(vec![p1.x, p1.y, p2.x, p2.y, p3.x, p3.y])
                }
                kurbo::PathEl::ClosePath => None,
                _ => panic!("Unsupported element type: {:?}", el),
            })
            .flatten()
            .collect();

        // Define a constraint: x0 > 0
        let cons: Vec<&dyn Func<()>> = vec![];
        // /        let cstr1 = |x: &[f64], _u: &mut ()| x.iter().cons.push(&cstr1);

        // Define a stopping criterion on objective function change
        let stop_tol = StopTols {
            ftol_rel: 1e-6,
            ..StopTols::default()
        };

        let bounds = xinit.iter().map(|v| (v - 50., v + 50.)).collect::<Vec<_>>();

        println!("xinit = {:?}", xinit);
        match minimize(
            |x: &[f64], data: &mut ()| self.cost(x, data),
            &xinit,
            &bounds,
            &cons,
            (),
            200,
            RhoBeg::All(0.5),
            Some(stop_tol),
        ) {
            Ok((status, x_opt, y_opt)) => {
                println!("status = {:?}", status);
                println!("x_opt = {:?}", x_opt);
                println!("y_opt = {}", y_opt);

                copy_control_points(&x_opt, self.path);
                println!("new perimeter: {}", self.path.perimeter(0.01));
                println!("new area: {}", self.path.area());
            }
            Err((e, _, _)) => println!("Optim error: {:?}", e),
        }
    }
}

fn copy_control_points(points: &[f64], path: &mut kurbo::BezPath) {
    let mut i = 0;
    for el in path.elements_mut() {
        match el {
            kurbo::PathEl::MoveTo(_) => {}
            kurbo::PathEl::CurveTo(p1, p2, p3) => {
                p1.x = points[i];
                p1.y = points[i + 1];
                p2.x = points[i + 2];
                p2.y = points[i + 3];
                p3.x = points[i + 4];
                p3.y = points[i + 5];
                i += 6;
            }
            kurbo::PathEl::ClosePath => {}
            _ => panic!("Unsupported element type: {:?}", el),
        }
    }
}
