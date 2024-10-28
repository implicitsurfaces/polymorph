import { test, describe } from "vitest";

import { ex } from "./test-utils";
import {
  angle_from_deg,
  angle_from_rad,
  as_vec,
  vec_from_cartesian_coords,
  vec_from_polar_coords,
} from "./geom";

describe("Angle", () => {
  test("angle from rad", () => {
    ex(angle_from_rad(0).cos()).toBeCloseTo(1);
    ex(angle_from_rad(Math.PI).cos()).toBeCloseTo(-1);
    ex(angle_from_rad(Math.PI / 2).cos()).toBeCloseTo(0);
    ex(angle_from_rad(0).sin()).toBeCloseTo(0);
    ex(angle_from_rad(Math.PI).sin()).toBeCloseTo(0);
    ex(angle_from_rad(Math.PI / 2).sin()).toBeCloseTo(1);
  });

  test("angle from deg", () => {
    ex(angle_from_deg(0).cos()).toBeCloseTo(1);
    ex(angle_from_deg(180).cos()).toBeCloseTo(-1);
    ex(angle_from_deg(90).cos()).toBeCloseTo(0);
    ex(angle_from_deg(45).cos()).toBeCloseTo(Math.sqrt(2) / 2);
    ex(angle_from_deg(45).sin()).toBeCloseTo(Math.sqrt(2) / 2);
  });

  test("as_deg", () => {
    ex(angle_from_rad(0).as_deg()).toBeCloseTo(0);
    ex(angle_from_rad(Math.PI).as_deg()).toBeCloseTo(180);
    ex(angle_from_rad(Math.PI / 2).as_deg()).toBeCloseTo(90);
    ex(angle_from_rad(Math.PI / 4).as_deg()).toBeCloseTo(45);
    ex(angle_from_deg(123).as_deg()).toBeCloseTo(123);
    ex(angle_from_deg(-123).as_deg()).toBeCloseTo(-123);
  });

  test("as_rad", () => {
    ex(angle_from_rad(0).as_rad()).toBeCloseTo(0);
    ex(angle_from_rad(Math.PI).as_rad()).toBeCloseTo(Math.PI);
    ex(angle_from_rad(1.1).as_rad()).toBeCloseTo(1.1);
    ex(angle_from_rad(-1.1).as_rad()).toBeCloseTo(-1.1);
  });

  function _d(x: number) {
    return angle_from_deg(x);
  }

  test("add", () => {
    ex(_d(0).add(_d(0)).as_deg()).toBeCloseTo(0);
    ex(_d(3).add(_d(120)).as_deg()).toBeCloseTo(123);
    ex(_d(90).add(_d(90)).as_deg()).toBeCloseTo(180);
    ex(_d(90).add(_d(-90)).as_deg()).toBeCloseTo(0);
    ex(_d(5).add(_d(50)).as_deg()).toBeCloseTo(55);
    ex(_d(250).add(_d(100)).as_deg()).toBeCloseTo(-10);
    ex(_d(250).add(_d(200)).as_deg()).toBeCloseTo(90);
  });

  test("sub", () => {
    ex(_d(0).sub(_d(0)).as_deg()).toBeCloseTo(0);
    ex(_d(120).sub(_d(3)).as_deg()).toBeCloseTo(117);
    ex(_d(180).sub(_d(90)).as_deg()).toBeCloseTo(90);
    ex(_d(0).sub(_d(90)).as_deg()).toBeCloseTo(-90);
    ex(_d(50).sub(_d(5)).as_deg()).toBeCloseTo(45);
    ex(_d(100).sub(_d(250)).as_deg()).toBeCloseTo(-150);
    ex(_d(-160).sub(_d(250)).as_deg()).toBeCloseTo(-50);
    ex(_d(-160).sub(_d(30)).as_deg()).toBeCloseTo(170);
  });

  test("double", () => {
    ex(_d(0).double().as_deg()).toBeCloseTo(0);
    ex(_d(-0).double().as_deg()).toBeCloseTo(0);
    ex(_d(90).double().as_deg()).toBeCloseTo(180);
    ex(_d(180).double().as_deg()).toBeCloseTo(0);
    ex(_d(270).double().as_deg()).toBeCloseTo(180);
    ex(_d(360).double().as_deg()).toBeCloseTo(0);

    ex(_d(40).double().as_deg()).toBeCloseTo(80);
    ex(_d(120).double().as_deg()).toBeCloseTo(-120);
    ex(_d(200).double().as_deg()).toBeCloseTo(40);
    ex(_d(300).double().as_deg()).toBeCloseTo(-120);
    ex(_d(-30).double().as_deg()).toBeCloseTo(-60);
    ex(_d(-100).double().as_deg()).toBeCloseTo(160);
  });

  test("half", () => {
    ex(_d(0).half().as_deg()).toBeCloseTo(0);
    ex(_d(-0).half().as_deg()).toBeCloseTo(0);
    ex(_d(90).half().as_deg()).toBeCloseTo(45);
    ex(_d(180).half().as_deg()).toBeCloseTo(90);
    ex(_d(120).half().as_deg()).toBeCloseTo(60);
    ex(_d(200).half().as_deg()).toBeCloseTo(100);
    ex(_d(300).half().as_deg()).toBeCloseTo(150);
    ex(_d(-30).half().as_deg()).toBeCloseTo(165);
    ex(_d(-100).half().as_deg()).toBeCloseTo(130);
  });

  test("perp", () => {
    ex(_d(0).perp().as_deg()).toBeCloseTo(90);
    ex(_d(-0).perp().as_deg()).toBeCloseTo(90);
    ex(_d(90).perp().as_deg()).toBeCloseTo(180);
    ex(_d(180).perp().as_deg()).toBeCloseTo(-90);
    ex(_d(120).perp().as_deg()).toBeCloseTo(-150);
    ex(_d(200).perp().as_deg()).toBeCloseTo(-70);
    ex(_d(300).perp().as_deg()).toBeCloseTo(30);
    ex(_d(-30).perp().as_deg()).toBeCloseTo(60);
    ex(_d(-100).perp().as_deg()).toBeCloseTo(-10);
  });

  test("opposite", () => {
    ex(_d(0).opposite().as_deg()).toBeCloseTo(-180);
    ex(_d(-0).opposite().as_deg()).toBeCloseTo(180);
    ex(_d(90).opposite().as_deg()).toBeCloseTo(-90);
    ex(_d(180).opposite().as_deg()).toBeCloseTo(0);
    ex(_d(270).opposite().as_deg()).toBeCloseTo(90);
    ex(_d(360).opposite().as_deg()).toBeCloseTo(180);

    ex(_d(50).opposite().as_deg()).toBeCloseTo(-130);
    ex(_d(150).opposite().as_deg()).toBeCloseTo(-30);
    ex(_d(200).opposite().as_deg()).toBeCloseTo(20);
    ex(_d(300).opposite().as_deg()).toBeCloseTo(120);

    ex(_d(-30).opposite().as_deg()).toBeCloseTo(150);
    ex(_d(-160).opposite().as_deg()).toBeCloseTo(20);
  });

  test("neg", () => {
    ex(_d(0).neg().as_deg()).toBeCloseTo(0);
    ex(_d(-0).neg().as_deg()).toBeCloseTo(0);
    ex(_d(90).neg().as_deg()).toBeCloseTo(-90);
    ex(_d(180).neg().as_deg()).toBeCloseTo(-180);
    ex(_d(270).neg().as_deg()).toBeCloseTo(90);
    ex(_d(360).neg().as_deg()).toBeCloseTo(0);
    ex(_d(50).neg().as_deg()).toBeCloseTo(-50);
    ex(_d(150).neg().as_deg()).toBeCloseTo(-150);
    ex(_d(200).neg().as_deg()).toBeCloseTo(160);
    ex(_d(300).neg().as_deg()).toBeCloseTo(60);
    ex(_d(-30).neg().as_deg()).toBeCloseTo(30);
    ex(_d(-160).neg().as_deg()).toBeCloseTo(160);
  });

  test("as_sort_value", () => {
    const angles = [
      0, 90, 180, 270, 360, 45, 135, 225, 315, 30, 120, 210, 300, 60, 150, 240,
      330, 15, 105, 195, 285, 75, 165, 255, 345,
    ];

    const cartesianProduct = angles.flatMap((a) => angles.map((b) => [a, b]));

    cartesianProduct.forEach(([a, b]) => {
      ex(_d(a).as_sort_value().sub(_d(b).as_sort_value()).sign()).toBeCloseTo(
        Math.sign(a - b),
      );
    });
  });
});

describe("Vec2", () => {
  test("add", () => {
    const v = as_vec(1, 2).add(as_vec(3, 4));
    ex(v.x).toBeCloseTo(4);
    ex(v.y).toBeCloseTo(6);
  });

  test("sub", () => {
    const v = as_vec(1, 2).sub(as_vec(3, 4));
    ex(v.x).toBeCloseTo(-2);
    ex(v.y).toBeCloseTo(-2);
  });

  test("scale", () => {
    const v = as_vec(1, 2).scale(3);
    ex(v.x).toBeCloseTo(3);
    ex(v.y).toBeCloseTo(6);
  });

  test("dot", () => {
    const v = as_vec(1, 2).dot(as_vec(3, 4));
    ex(v).toBeCloseTo(11);
  });

  test("cross", () => {
    const v = as_vec(1, 2).cross(as_vec(3, 4));
    ex(v).toBeCloseTo(-2);
  });

  test("norm", () => {
    const v = as_vec(3, 4).norm();
    ex(v).toBeCloseTo(5);
  });

  test("normalize", () => {
    ex(as_vec(3, 4).normalize().norm()).toBeCloseTo(1);
  });

  test("as_angle", () => {
    ex(as_vec(3, 3).as_angle().as_deg()).toBeCloseTo(45);
    ex(as_vec(3, -3).as_angle().as_deg()).toBeCloseTo(-45);
    ex(as_vec(-3, 3).as_angle().as_deg()).toBeCloseTo(135);
    ex(as_vec(-3, -3).as_angle().as_deg()).toBeCloseTo(-135);
  });

  test("vec_from_polar_coords", () => {
    const v = vec_from_polar_coords(3, angle_from_deg(33));
    ex(v.as_angle().as_deg()).toBeCloseTo(33);
    ex(v.norm()).toBeCloseTo(3);
  });

  test("rotate", () => {
    const v = vec_from_polar_coords(3, angle_from_deg(33)).rotate(
      angle_from_deg(22),
    );
    ex(v.as_angle().as_deg()).toBeCloseTo(55);

    const v2 = vec_from_polar_coords(3, angle_from_deg(33)).rotate(
      angle_from_deg(-55),
    );
    ex(v2.as_angle().as_deg()).toBeCloseTo(-22);
  });
});
