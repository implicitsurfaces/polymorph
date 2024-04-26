use cobyla::{minimize, FailStatus, Func, RhoBeg, StopTols};
use kurbo::{BezPath, Circle, Point, Shape};
use rand::distributions::{Distribution, Uniform}; // 0.6.5

use std::f64::consts::PI;
use std::f64::INFINITY;
use std::io::Write;

use std::fs::File;

fn save_svg(path: &str, figures: &Vec<BezPath>) {
    // we write all the paths in a single svg
    let mut string = String::new();
    string.push_str(r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="-1.5 -1.5 3 3" fill="none" stroke="black" stroke-width="0.2%" vector-effect="non-scaling-stroke">\n"#);

    for figure in figures {
        string.push_str("<path d=\"");
        string.push_str(&figure.to_svg());
        string.push_str("\" />\n")
    }
    string.push_str("</svg>");

    let mut file = File::create(path).unwrap();
    file.write_all(string.as_bytes()).unwrap();
}

fn three_points_circle_center(p1: Point, p2: Point, p3: Point) -> Point {
    let mid_point_1_2 = p1.midpoint(p2);
    let mid_point_2_3 = p2.midpoint(p3);

    let slope_1_2 = (p2.y - p1.y) / (p2.x - p1.x);
    let slope_2_3 = (p3.y - p2.y) / (p3.x - p2.x);

    let slope_perpendicular_1_2 = -1. / slope_1_2;
    let slope_perpendicular_2_3 = -1. / slope_2_3;

    let intercept_1_2 = mid_point_1_2.y - slope_perpendicular_1_2 * mid_point_1_2.x;
    let intercept_2_3 = mid_point_2_3.y - slope_perpendicular_2_3 * mid_point_2_3.x;

    return Point::new(
        (intercept_2_3 - intercept_1_2) / (slope_perpendicular_1_2 - slope_perpendicular_2_3),
        slope_perpendicular_1_2 * (intercept_2_3 - intercept_1_2)
            / (slope_perpendicular_1_2 - slope_perpendicular_2_3)
            + intercept_1_2,
    );
}

fn tangent_from_points(p1: Point, p2: Point, p3: Point) -> f64 {
    let center = three_points_circle_center(p1, p2, p3);
    return (center - p2).atan2() - PI / 2.;
}

fn default_tangents_from_points(points: &Vec<Point>) -> Vec<f64> {
    return points
        .iter()
        .enumerate()
        .map(|(i, _)| {
            tangent_from_points(
                points[(i + points.len() - 1) % points.len()],
                points[i],
                points[(i + 1) % points.len()],
            )
        })
        .collect();
}

fn default_magnitudes_from_points(points: &Vec<Point>) -> Vec<f64> {
    let mut magnitudes = Vec::new();
    for i in 0..points.len() {
        if i == (points.len() - 1) {
            magnitudes.push(points[i].distance(points[0]) / 3.);
        } else {
            magnitudes.push(points[i].distance(points[i + 1]) / 3.);
        }
    }
    magnitudes
}

struct MyShape {
    points: Vec<Point>,
    tangents: Vec<f64>,
    magnitudes: Vec<f64>,
}

impl MyShape {
    fn new(points: &Vec<Point>, tangents: &Vec<f64>, magnitudes: &Vec<f64>) -> Self {
        MyShape {
            points: points.clone(),
            tangents: tangents.clone(),
            magnitudes: magnitudes.clone(),
        }
    }

    fn new_with_defaults(points: &Vec<Point>) -> Self {
        let tangents = default_tangents_from_points(points);
        let magnitudes = default_magnitudes_from_points(points);
        MyShape::new(points, &tangents, &magnitudes)
    }

    fn start_control_point(&self, curve_index: usize) -> Point {
        let cos = self.tangents[curve_index].cos();
        let sin = self.tangents[curve_index].sin();

        return Point::new(
            self.points[curve_index].x + cos * self.magnitudes[curve_index],
            self.points[curve_index].y + sin * self.magnitudes[curve_index],
        );
    }

    fn end_control_point(&self, curve_index: usize) -> Point {
        let index = (curve_index + 1) % self.points.len();

        let cos = self.tangents[index].cos();
        let sin = self.tangents[index].sin();

        return Point::new(
            self.points[index].x - cos * self.magnitudes[curve_index],
            self.points[index].y - sin * self.magnitudes[curve_index],
        );
    }

    fn figure(&self) -> BezPath {
        let mut fig = BezPath::new();

        fig.move_to(self.points[0]);

        for i in 1..(self.points.len() + 1) {
            fig.curve_to(
                self.start_control_point(i - 1),
                self.end_control_point(i - 1),
                self.points[i % self.points.len()],
            );
        }

        fig
    }
}

fn generate_initial_points(size: usize) -> Vec<Point> {
    let step = Uniform::new(0., PI * 2.);
    let mut rng = rand::thread_rng();
    let mut angles: Vec<f64> = step.sample_iter(&mut rng).take(size).collect::<Vec<f64>>();

    angles.sort_by(|a, b| (a).partial_cmp(b).unwrap());

    let points: Vec<Point> = angles
        .iter()
        .map(|angle| Point::new(angle.cos(), angle.sin()))
        .collect();

    points
}

struct Problem {
    points: Vec<Point>,
}

impl Problem {
    fn new(size: usize) -> Self {
        Problem {
            points: generate_initial_points(size),
        }
    }

    fn cost(&self, x: &[f64], _data: &mut ()) -> f64 {
        let tangents = x[0..self.points.len()].to_vec();
        let magnitudes = x[self.points.len()..].to_vec();

        if tangents.iter().any(|&v| v.abs() > 2. * PI) {
            return INFINITY;
        }

        if magnitudes.iter().any(|&v| v < 0.) {
            return INFINITY;
        }

        let shape = MyShape::new(&self.points, &tangents, &magnitudes);

        let figure = shape.figure();

        let len = figure.perimeter(0.01);
        let area = figure.area();

        let length_err = (len.ln()).powi(2) / 2.;
        let surface_err = (area - PI).powi(2) / 0.2;

        return length_err + surface_err;
    }

    fn optimize(self) -> Result<MyShape, FailStatus> {
        let mut xinit: Vec<f64> = default_tangents_from_points(&self.points);
        xinit.extend(default_magnitudes_from_points(&self.points));

        let bounds = self
            .points
            .iter()
            .map(|_| (-2. * PI, 2. * PI))
            .chain(self.points.iter().map(|_| (0., INFINITY)))
            .collect::<Vec<_>>();

        let cons: Vec<&dyn Func<()>> = vec![];

        let stop_tol = StopTols {
            ftol_rel: 1e-6,
            ..StopTols::default()
        };

        let results = minimize(
            |x: &[f64], data: &mut ()| self.cost(x, data),
            &xinit,
            &bounds,
            &cons,
            (),
            200,
            RhoBeg::All(0.5),
            Some(stop_tol),
        );

        match results {
            Ok((_, x_opt, _)) => {
                let tangents = x_opt[0..self.points.len()].to_vec();
                let magnitudes = x_opt[self.points.len()..].to_vec();

                return Ok(MyShape::new(&self.points, &tangents, &magnitudes));
            }
            Err((e, _, _)) => {
                return Err(e);
            }
        }
    }
}

fn main() {
    let result = Problem::new(6).optimize();

    match result {
        Err(e) => {
            println!("Optimization failed: {:?}", e);
        }

        Ok(shape) => {
            let figure = shape.figure();
            println!(
                "perimeter: {}\narea: {}",
                figure.perimeter(0.001),
                figure.area().abs()
            );

            let mut output_shapes: Vec<BezPath> = vec![figure];

            let point_circles = shape
                .points
                .iter()
                .map(|p| (Circle::new(p.clone(), 0.02).to_path(0.001)));

            output_shapes.extend(point_circles);

            save_svg("output.svg", &output_shapes);

            return;
        }
    }
}
