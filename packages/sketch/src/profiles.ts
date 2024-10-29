import Point from "./geom";
import { Num, as_num } from "./num";
import { hypot, max } from "./num-ops";

export class Circle {
  readonly radius: Num;
  constructor(radius: Num | number) {
    this.radius = as_num(radius);
  }

  distance_to(point: Point): Num {
    return point.vec_from_origin().norm().sub(this.radius);
  }
}

export class Box {
  readonly width: Num;
  readonly height: Num;
  constructor(width: Num | number, height: Num | number) {
    this.width = as_num(width);
    this.height = as_num(height);
  }

  distance_to(point: Point): Num {
    const half_width = this.width.div(2);
    const half_height = this.height.div(2);

    const q_x = point.x.smoothabs().sub(half_width);
    const q_y = point.y.smoothabs().sub(half_height);

    return hypot(q_x.softplus(), q_y.softplus()).add(max(q_x, q_y).softminus());
  }
}

export class TopHalfPlane {
  distance_to(point: Point): Num {
    return point.y;
  }
}

export class BottomHalfPlane {
  distance_to(point: Point): Num {
    return point.y.neg();
  }
}

export class LeftHalfPlane {
  distance_to(point: Point): Num {
    return point.x;
  }
}

export class RightHalfPlane {
  distance_to(point: Point): Num {
    return point.x.neg();
  }
}
