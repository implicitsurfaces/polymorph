#![warn(clippy::all, rust_2018_idioms)]

use geo::*;

mod app;
pub use app::App;

pub trait ToEguiShape {
    fn to_egui_shape(&self, color: egui::Color32) -> egui::Shape;
}

impl ToEguiShape for Polygon {
    fn to_egui_shape(&self, color: egui::Color32) -> egui::Shape {
        let points = self
            .exterior()
            .points()
            .map(|p| egui::pos2(p.x() as f32, p.y() as f32))
            .collect();

        egui::Shape::convex_polygon(points, color, egui::Stroke::new(1.0, egui::Color32::BLACK))
    }
}

impl ToEguiShape for Point {
    fn to_egui_shape(&self, color: egui::Color32) -> egui::Shape {
        let point = egui::pos2(self.x() as f32, self.y() as f32);
        egui::Shape::circle_filled(point, 2.0, color)
    }
}

impl ToEguiShape for Line {
    fn to_egui_shape(&self, color: egui::Color32) -> egui::Shape {
        egui::Shape::line_segment(
            [
                egui::pos2(self.start.x as f32, self.start.y as f32),
                egui::pos2(self.end.x as f32, self.end.y as f32),
            ],
            egui::Stroke::new(1.0, color),
        )
    }
}
