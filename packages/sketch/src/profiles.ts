import { Point } from "./geom";
import { Num, asNum } from "./num";
import { hypot, max } from "./num-ops";

export class Circle {
  readonly radius: Num;
  constructor(radius: Num | number) {
    this.radius = asNum(radius);
  }

  distanceTo(point: Point): Num {
    return point.vecFromOrigin().norm().sub(this.radius);
  }
}

export class Box {
  readonly width: Num;
  readonly height: Num;
  constructor(width: Num | number, height: Num | number) {
    this.width = asNum(width);
    this.height = asNum(height);
  }

  distanceTo(point: Point): Num {
    const halfWidth = this.width.div(2);
    const halfHeight = this.height.div(2);

    const qX = point.x.smoothabs().sub(halfWidth);
    const qY = point.y.smoothabs().sub(halfHeight);

    return hypot(qX.softplus(), qY.softplus()).add(max(qX, qY).softminus());
  }
}

export class TopHalfPlane {
  distanceTo(point: Point): Num {
    return point.y;
  }
}

export class BottomHalfPlane {
  distanceTo(point: Point): Num {
    return point.y.neg();
  }
}

export class LeftHalfPlane {
  distanceTo(point: Point): Num {
    return point.x;
  }
}

export class RightHalfPlane {
  distanceTo(point: Point): Num {
    return point.x.neg();
  }
}
