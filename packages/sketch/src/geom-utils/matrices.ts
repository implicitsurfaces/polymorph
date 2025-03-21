import { Num, ONE, TWO, ZERO } from "../num";
import { ifTruthyElse } from "../num-ops";

export class ColVec2 {
  constructor(
    public readonly x1: Num,
    public readonly x2: Num,
  ) {}

  transpose(): RowVec2 {
    return new RowVec2(this.x1, this.x2);
  }
}

export class RowVec2 {
  constructor(
    public readonly x1: Num,
    public readonly x2: Num,
  ) {}

  transpose(): ColVec2 {
    return new ColVec2(this.x1, this.x2);
  }

  dot(other: ColVec2): Num {
    return this.x1.mul(other.x1).add(this.x2.mul(other.x2));
  }

  product(matrix: Matrix2x2): RowVec2 {
    return new RowVec2(
      this.x1.mul(matrix.x11).add(this.x2.mul(matrix.x21)),
      this.x1.mul(matrix.x12).add(this.x2.mul(matrix.x22)),
    );
  }
}

function det22(a: Num, b: Num, c: Num, d: Num): Num {
  return a.mul(d).sub(b.mul(c));
}

export class Matrix2x2 {
  constructor(
    public readonly x11: Num,
    public readonly x12: Num,
    public readonly x21: Num,
    public readonly x22: Num,
  ) {}

  transpose(): Matrix2x2 {
    return new Matrix2x2(this.x11, this.x21, this.x12, this.x22);
  }

  mul(other: Matrix2x2): Matrix2x2 {
    return new Matrix2x2(
      this.x11.mul(other.x11).add(this.x12.mul(other.x21)),
      this.x11.mul(other.x12).add(this.x12.mul(other.x22)),
      this.x21.mul(other.x11).add(this.x22.mul(other.x21)),
      this.x21.mul(other.x12).add(this.x22.mul(other.x22)),
    );
  }

  add(other: Matrix2x2): Matrix2x2 {
    return new Matrix2x2(
      this.x11.add(other.x11),
      this.x12.add(other.x12),
      this.x21.add(other.x21),
      this.x22.add(other.x22),
    );
  }

  sub(other: Matrix2x2): Matrix2x2 {
    return new Matrix2x2(
      this.x11.sub(other.x11),
      this.x12.sub(other.x12),
      this.x21.sub(other.x21),
      this.x22.sub(other.x22),
    );
  }

  scale(scalar: Num): Matrix2x2 {
    return new Matrix2x2(
      this.x11.mul(scalar),
      this.x12.mul(scalar),
      this.x21.mul(scalar),
      this.x22.mul(scalar),
    );
  }

  product(colVec: ColVec2): ColVec2 {
    return new ColVec2(
      this.x11.mul(colVec.x1).add(this.x12.mul(colVec.x2)),
      this.x21.mul(colVec.x1).add(this.x22.mul(colVec.x2)),
    );
  }

  det(): Num {
    return det22(this.x11, this.x12, this.x21, this.x22);
  }

  trace(): Num {
    return this.x11.add(this.x22);
  }

  inverse(): Matrix2x2 {
    const det = this.det();
    return new Matrix2x2(
      this.x22.div(det),
      this.x12.div(det).neg(),
      this.x21.div(det).neg(),
      this.x11.div(det),
    );
  }

  eigenvalues(): [Num, Num] {
    const trace = this.trace();
    const mean = trace.div(TWO);
    const product = this.det();

    const discriminant = mean.square().sub(product).sqrt();

    return [mean.sub(discriminant), mean.add(discriminant)];
  }

  eigenvector(lambda: Num): ColVec2 {
    const x = ifTruthyElse(this.x12, this.x12, lambda.sub(this.x22));
    const y = ifTruthyElse(this.x12, lambda.sub(this.x11), this.x21);

    const isNonDiagonalMatrix = this.x12.or(this.x21);
    const aCase = this.x11.equals(lambda);

    const diagValueX = ifTruthyElse(aCase, ONE, ZERO);
    const diagValueY = ifTruthyElse(aCase, ZERO, ONE);

    return new ColVec2(
      ifTruthyElse(isNonDiagonalMatrix, x, diagValueX),
      ifTruthyElse(isNonDiagonalMatrix, y, diagValueY),
    );
  }

  eigenvectors(): [ColVec2, ColVec2] {
    const [lambda1, lambda2] = this.eigenvalues();
    return [this.eigenvector(lambda1), this.eigenvector(lambda2)];
  }
}

export class RowVec3 {
  constructor(
    public readonly x1: Num,
    public readonly x2: Num,
    public readonly x3: Num,
  ) {}

  transpose(): ColVec3 {
    return new ColVec3(this.x1, this.x2, this.x3);
  }

  dot(other: ColVec3): Num {
    return this.x1
      .mul(other.x1)
      .add(this.x2.mul(other.x2))
      .add(this.x3.mul(other.x3));
  }

  product(matrix: Matrix3x3): RowVec3 {
    return new RowVec3(
      this.x1
        .mul(matrix.x11)
        .add(this.x2.mul(matrix.x21))
        .add(this.x3.mul(matrix.x31)),
      this.x1
        .mul(matrix.x12)
        .add(this.x2.mul(matrix.x22))
        .add(this.x3.mul(matrix.x32)),
      this.x1
        .mul(matrix.x13)
        .add(this.x2.mul(matrix.x23))
        .add(this.x3.mul(matrix.x33)),
    );
  }
}

export class ColVec3 {
  constructor(
    public readonly x1: Num,
    public readonly x2: Num,
    public readonly x3: Num,
  ) {}

  transpose(): RowVec3 {
    return new RowVec3(this.x1, this.x2, this.x3);
  }
}

export class Matrix3x3 {
  constructor(
    public readonly x11: Num,
    public readonly x12: Num,
    public readonly x13: Num,
    public readonly x21: Num,
    public readonly x22: Num,
    public readonly x23: Num,
    public readonly x31: Num,
    public readonly x32: Num,
    public readonly x33: Num,
  ) {}

  transpose(): Matrix3x3 {
    return new Matrix3x3(
      this.x11,
      this.x21,
      this.x31,
      this.x12,
      this.x22,
      this.x32,
      this.x13,
      this.x23,
      this.x33,
    );
  }

  mul(other: Matrix3x3): Matrix3x3 {
    return new Matrix3x3(
      this.x11
        .mul(other.x11)
        .add(this.x12.mul(other.x21))
        .add(this.x13.mul(other.x31)),
      this.x11
        .mul(other.x12)
        .add(this.x12.mul(other.x22))
        .add(this.x13.mul(other.x32)),
      this.x11
        .mul(other.x13)
        .add(this.x12.mul(other.x23))
        .add(this.x13.mul(other.x33)),
      this.x21
        .mul(other.x11)
        .add(this.x22.mul(other.x21))
        .add(this.x23.mul(other.x31)),
      this.x21
        .mul(other.x12)
        .add(this.x22.mul(other.x22))
        .add(this.x23.mul(other.x32)),
      this.x21
        .mul(other.x13)
        .add(this.x22.mul(other.x23))
        .add(this.x23.mul(other.x33)),
      this.x31
        .mul(other.x11)
        .add(this.x32.mul(other.x21))
        .add(this.x33.mul(other.x31)),
      this.x31
        .mul(other.x12)
        .add(this.x32.mul(other.x22))
        .add(this.x33.mul(other.x32)),
      this.x31
        .mul(other.x13)
        .add(this.x32.mul(other.x23))
        .add(this.x33.mul(other.x33)),
    );
  }

  add(other: Matrix3x3): Matrix3x3 {
    return new Matrix3x3(
      this.x11.add(other.x11),
      this.x12.add(other.x12),
      this.x13.add(other.x13),
      this.x21.add(other.x21),
      this.x22.add(other.x22),
      this.x23.add(other.x23),
      this.x31.add(other.x31),
      this.x32.add(other.x32),
      this.x33.add(other.x33),
    );
  }

  sub(other: Matrix3x3): Matrix3x3 {
    return new Matrix3x3(
      this.x11.sub(other.x11),
      this.x12.sub(other.x12),
      this.x13.sub(other.x13),
      this.x21.sub(other.x21),
      this.x22.sub(other.x22),
      this.x23.sub(other.x23),
      this.x31.sub(other.x31),
      this.x32.sub(other.x32),
      this.x33.sub(other.x33),
    );
  }

  scale(scalar: Num): Matrix3x3 {
    return new Matrix3x3(
      this.x11.mul(scalar),
      this.x12.mul(scalar),
      this.x13.mul(scalar),
      this.x21.mul(scalar),
      this.x22.mul(scalar),
      this.x23.mul(scalar),
      this.x31.mul(scalar),
      this.x32.mul(scalar),
      this.x33.mul(scalar),
    );
  }

  product(colVec: ColVec3): ColVec3 {
    return new ColVec3(
      this.x11
        .mul(colVec.x1)
        .add(this.x12.mul(colVec.x2))
        .add(this.x13.mul(colVec.x3)),
      this.x21
        .mul(colVec.x1)
        .add(this.x22.mul(colVec.x2))
        .add(this.x23.mul(colVec.x3)),
      this.x31
        .mul(colVec.x1)
        .add(this.x32.mul(colVec.x2))
        .add(this.x33.mul(colVec.x3)),
    );
  }

  det(): Num {
    return det22(this.x22, this.x23, this.x32, this.x33)
      .mul(this.x11)
      .sub(det22(this.x12, this.x13, this.x32, this.x33).mul(this.x21))
      .add(det22(this.x12, this.x13, this.x22, this.x23).mul(this.x31));
  }

  trace(): Num {
    return this.x11.add(this.x22).add(this.x33);
  }

  inverse(): Matrix3x3 {
    const det = this.det();
    return new Matrix3x3(
      det22(this.x22, this.x23, this.x32, this.x33).div(det), // M11
      det22(this.x21, this.x23, this.x31, this.x33).div(det).neg(), // M12
      det22(this.x21, this.x22, this.x31, this.x32).div(det), // M13
      det22(this.x12, this.x13, this.x32, this.x33).div(det).neg(), // M21
      det22(this.x11, this.x13, this.x31, this.x33).div(det), // M22
      det22(this.x11, this.x12, this.x31, this.x32).div(det).neg(), // M23
      det22(this.x12, this.x13, this.x22, this.x23).div(det), // M31
      det22(this.x11, this.x13, this.x21, this.x23).div(det).neg(), // M32
      det22(this.x11, this.x12, this.x21, this.x22).div(det), // M33
    );
  }
}

export const IDENTITY_MATRIX_2x2 = new Matrix2x2(ONE, ZERO, ZERO, ONE);

export const IDENTITY_MATRIX_3x3 = new Matrix3x3(
  ONE,
  ZERO,
  ZERO,
  ZERO,
  ONE,
  ZERO,
  ZERO,
  ZERO,
  ONE,
);
