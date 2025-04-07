export class Primitive2D {
  constructor() {}
}

class Line2D extends Primitive2D {
  constructor() {
    super();
  }
}

export class HalfPlane2D extends Primitive2D {
  constructor() {
    super();
  }
}

export class Conic2D extends Primitive2D {
  constructor() {
    super();
  }
}

type Curve2D = Line2D | Conic2D;

export class Segment2D extends Primitive2D {
  constructor(
    public readonly curve: Curve2D,
    public readonly section: HalfPlane2D[],
  ) {
    super();
  }
}

export class Primitive3D {
  constructor() {}
}

export class Line3D extends Primitive3D {
  constructor(
    public readonly line: Line2D,
    public readonly plane: Plane,
  ) {
    super();
  }
}

export class Conic3D extends Primitive3D {
  constructor(
    public readonly conic: Conic3D,
    public readonly plane: Plane,
  ) {
    super();
  }
}

type Curve3D = Line3D | Conic3D;

export class Plane extends Primitive3D {
  constructor() {
    super();
  }
}

export class HalfSpace3D extends Primitive3D {
  constructor() {
    super();
  }
}

export class Cone extends Primitive3D {
  constructor(
    public readonly baseCurve: Curve3D,
    public readonly apex: Point3D,
  ) {
    super();
  }
}

export class Cylinder extends Primitive3D {
  constructor(
    public readonly baseCurve: Curve3D,
    public readonly height: number,
  ) {
    super();
  }
}

type Surface3D = Plane | Cone | Cylinder;

export class Patch extends Primitive3D {
  constructor(
    public readonly baseSurface: Surface3D,
    public readonly section: HalfSpace3D[],
  ) {
    super();
  }
}
