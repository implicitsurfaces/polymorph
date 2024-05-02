#![warn(clippy::all, rust_2018_idioms)]

use geo::*;

mod app;
pub use app::App;

pub trait ToEguiShape {
    fn to_egui_shapes(&self, fill_color: egui::Color32) -> Vec<egui::Shape>;
}

impl ToEguiShape for Triangle {
    fn to_egui_shapes(&self, fill_color: egui::Color32) -> Vec<egui::Shape> {
        let points = self
            .coords_iter()
            .map(|p| egui::pos2(p.x as f32, p.y as f32))
            .collect();

        // Note: this assumes that a triangle should never be stroked.
        vec![egui::Shape::convex_polygon(
            points,
            fill_color,
            egui::Stroke::NONE,
        )]
    }
}

impl ToEguiShape for Polygon {
    fn to_egui_shapes(&self, fill_color: egui::Color32) -> Vec<egui::Shape> {
        let mut shapes: Vec<egui::Shape> = self
            .earcut_triangles()
            .iter()
            .map(|tri| {
                let points = tri
                    .coords_iter()
                    .map(|p| egui::pos2(p.x as f32, p.y as f32))
                    .collect();

                egui::Shape::convex_polygon(points, fill_color, egui::Stroke::new(1.0, fill_color))
            })
            .collect();
        let exterior_points = self
            .exterior()
            .coords_iter()
            .map(|p| egui::pos2(p.x as f32, p.y as f32))
            .collect();
        shapes.push(egui::Shape::line(
            exterior_points,
            egui::Stroke::new(1.0, egui::Color32::BLACK),
        ));
        shapes
    }
}

impl ToEguiShape for Point {
    fn to_egui_shapes(&self, fill_color: egui::Color32) -> Vec<egui::Shape> {
        let point = egui::pos2(self.x() as f32, self.y() as f32);
        vec![egui::Shape::circle_filled(point, 2.0, fill_color)]
    }
}

impl ToEguiShape for Line {
    fn to_egui_shapes(&self, fill_color: egui::Color32) -> Vec<egui::Shape> {
        vec![egui::Shape::line_segment(
            [
                egui::pos2(self.start.x as f32, self.start.y as f32),
                egui::pos2(self.end.x as f32, self.end.y as f32),
            ],
            egui::Stroke::new(1.0, fill_color),
        )]
    }
}
