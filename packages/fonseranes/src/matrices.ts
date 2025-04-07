export class ColVec3 {
  constructor(
    public readonly x1: number,
    public readonly x2: number,
    public readonly x3: number,
  ) {}

  get repr() {
    return `(
  ${this.x1}, 
  ${this.x2}, 
  ${this.x3}
)`;
  }

  add(other: ColVec3): ColVec3 {
    return new ColVec3(
      this.x1 + other.x1,
      this.x2 + other.x2,
      this.x3 + other.x3,
    );
  }

  sub(other: ColVec3): ColVec3 {
    return new ColVec3(
      this.x1 - other.x1,
      this.x2 - other.x2,
      this.x3 - other.x3,
    );
  }

  transpose(): RowVec3 {
    return new RowVec3(this.x1, this.x2, this.x3);
  }
}

export class RowVec3 {
  constructor(
    public readonly x1: number,
    public readonly x2: number,
    public readonly x3: number,
  ) {}

  get repr() {
    return `(${this.x1}, ${this.x2}, ${this.x3})`;
  }

  transpose(): ColVec3 {
    return new ColVec3(this.x1, this.x2, this.x3);
  }

  add(other: RowVec3): RowVec3 {
    return new RowVec3(
      this.x1 + other.x1,
      this.x2 + other.x2,
      this.x3 + other.x3,
    );
  }

  sub(other: RowVec3): RowVec3 {
    return new RowVec3(
      this.x1 - other.x1,
      this.x2 - other.x2,
      this.x3 - other.x3,
    );
  }

  dot(other: ColVec3): number {
    return this.x1 * other.x1 + this.x2 * other.x2 + this.x3 * other.x3;
  }

  product(matrix: Matrix3x3): RowVec3 {
    return new RowVec3(
      this.x1 * matrix.x11 + this.x2 * matrix.x21 + this.x3 * matrix.x31,
      this.x1 * matrix.x12 + this.x2 * matrix.x22 + this.x3 * matrix.x32,
      this.x1 * matrix.x13 + this.x2 * matrix.x23 + this.x3 * matrix.x33,
    );
  }
}

export class Matrix3x3 {
  constructor(
    public readonly x11: number,
    public readonly x12: number,
    public readonly x13: number,
    public readonly x21: number,
    public readonly x22: number,
    public readonly x23: number,
    public readonly x31: number,
    public readonly x32: number,
    public readonly x33: number,
  ) {}

  get repr() {
    return `(
  ${this.x11}, ${this.x12}, ${this.x13}, 
  ${this.x21}, ${this.x22}, ${this.x23}, 
  ${this.x31}, ${this.x32}, ${this.x33}
)`;
  }

  mul(other: Matrix3x3): Matrix3x3 {
    return new Matrix3x3(
      this.x11 * other.x11 + this.x12 * other.x21 + this.x13 * other.x31,
      this.x11 * other.x12 + this.x12 * other.x22 + this.x13 * other.x32,
      this.x11 * other.x13 + this.x12 * other.x23 + this.x13 * other.x33,
      this.x21 * other.x11 + this.x22 * other.x21 + this.x23 * other.x31,
      this.x21 * other.x12 + this.x22 * other.x22 + this.x23 * other.x32,
      this.x21 * other.x13 + this.x22 * other.x23 + this.x23 * other.x33,
      this.x31 * other.x11 + this.x32 * other.x21 + this.x33 * other.x31,
      this.x31 * other.x12 + this.x32 * other.x22 + this.x33 * other.x32,
      this.x31 * other.x13 + this.x32 * other.x23 + this.x33 * other.x33,
    );
  }

  add(other: Matrix3x3): Matrix3x3 {
    // prettier-ignore
    return new Matrix3x3(
      this.x11 + other.x11, this.x12 + other.x12, this.x13 + other.x13,
      this.x21 + other.x21, this.x22 + other.x22, this.x23 + other.x23,
      this.x31 + other.x31, this.x32 + other.x32, this.x33 + other.x33,
    );
  }

  sub(other: Matrix3x3): Matrix3x3 {
    // prettier-ignore
    return new Matrix3x3(
      this.x11 - other.x11, this.x12 - other.x12, this.x13 - other.x13,
      this.x21 - other.x21, this.x22 - other.x22, this.x23 - other.x23,
      this.x31 - other.x31, this.x32 - other.x32, this.x33 - other.x33,
    );
  }

  scale(scalar: number): Matrix3x3 {
    // prettier-ignore
    return new Matrix3x3(
      this.x11 * scalar, this.x12 * scalar, this.x13 * scalar,
      this.x21 * scalar, this.x22 * scalar, this.x23 * scalar,
      this.x31 * scalar, this.x32 * scalar, this.x33 * scalar,
    );
  }

  product(v: ColVec3): ColVec3 {
    return new ColVec3(
      this.x11 * v.x1 + this.x12 * v.x2 + this.x13 * v.x3,
      this.x21 * v.x1 + this.x22 * v.x2 + this.x23 * v.x3,
      this.x31 * v.x1 + this.x32 * v.x2 + this.x33 * v.x3,
    );
  }

  det(): number {
    return (
      this.x11 * (this.x22 * this.x33 - this.x23 * this.x32) -
      this.x12 * (this.x21 * this.x33 - this.x23 * this.x31) +
      this.x13 * (this.x21 * this.x32 - this.x22 * this.x31)
    );
  }

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

  inverse(): Matrix3x3 {
    const det = this.det();

    if (det === 0) {
      throw new Error("Matrix is singular and cannot be inverted");
    }

    return new Matrix3x3(
      (this.x22 * this.x33 - this.x23 * this.x32) / det,
      (this.x13 * this.x32 - this.x12 * this.x33) / det,
      (this.x12 * this.x23 - this.x13 * this.x22) / det,
      (this.x23 * this.x31 - this.x21 * this.x33) / det,
      (this.x11 * this.x33 - this.x13 * this.x31) / det,
      (this.x13 * this.x21 - this.x11 * this.x23) / det,
      (this.x21 * this.x32 - this.x22 * this.x31) / det,
      (this.x12 * this.x31 - this.x11 * this.x32) / det,
      (this.x11 * this.x22 - this.x12 * this.x21) / det,
    );
  }
}

export const diagonalMatrix3x3 = (
  x11: number,
  x22: number,
  x33: number,
): Matrix3x3 => {
  return new Matrix3x3(x11, 0, 0, 0, x22, 0, 0, 0, x33);
};
export const IDENTITY_MATRIX_3x3 = diagonalMatrix3x3(1, 1, 1);

export class ColVec4 {
  constructor(
    public readonly x1: number,
    public readonly x2: number,
    public readonly x3: number,
    public readonly x4: number,
  ) {}

  get repr() {
    return `(
  ${this.x1},
  ${this.x2},
  ${this.x3},
  ${this.x4}
)`;
  }

  add(other: ColVec4): ColVec4 {
    return new ColVec4(
      this.x1 + other.x1,
      this.x2 + other.x2,
      this.x3 + other.x3,
      this.x4 + other.x4,
    );
  }

  sub(other: ColVec4): ColVec4 {
    return new ColVec4(
      this.x1 - other.x1,
      this.x2 - other.x2,
      this.x3 - other.x3,
      this.x4 - other.x4,
    );
  }

  transpose(): RowVec4 {
    return new RowVec4(this.x1, this.x2, this.x3, this.x4);
  }
}

export class RowVec4 {
  constructor(
    public readonly x1: number,
    public readonly x2: number,
    public readonly x3: number,
    public readonly x4: number,
  ) {}

  get repr() {
    return `(${this.x1}, ${this.x2}, ${this.x3}, ${this.x4})`;
  }

  add(other: RowVec4): RowVec4 {
    return new RowVec4(
      this.x1 + other.x1,
      this.x2 + other.x2,
      this.x3 + other.x3,
      this.x4 + other.x4,
    );
  }

  sub(other: RowVec4): RowVec4 {
    return new RowVec4(
      this.x1 - other.x1,
      this.x2 - other.x2,
      this.x3 - other.x3,
      this.x4 - other.x4,
    );
  }

  transpose(): ColVec4 {
    return new ColVec4(this.x1, this.x2, this.x3, this.x4);
  }

  dot(other: ColVec4): number {
    return (
      this.x1 * other.x1 +
      this.x2 * other.x2 +
      this.x3 * other.x3 +
      this.x4 * other.x4
    );
  }

  product(matrix: Matrix4x4): RowVec4 {
    return new RowVec4(
      this.x1 * matrix.x11 +
        this.x2 * matrix.x21 +
        this.x3 * matrix.x31 +
        this.x4 * matrix.x41,
      this.x1 * matrix.x12 +
        this.x2 * matrix.x22 +
        this.x3 * matrix.x32 +
        this.x4 * matrix.x42,
      this.x1 * matrix.x13 +
        this.x2 * matrix.x23 +
        this.x3 * matrix.x33 +
        this.x4 * matrix.x43,
      this.x1 * matrix.x14 +
        this.x2 * matrix.x24 +
        this.x3 * matrix.x34 +
        this.x4 * matrix.x44,
    );
  }
}

export class Matrix4x4 {
  constructor(
    public readonly x11: number,
    public readonly x12: number,
    public readonly x13: number,
    public readonly x14: number,
    public readonly x21: number,
    public readonly x22: number,
    public readonly x23: number,
    public readonly x24: number,
    public readonly x31: number,
    public readonly x32: number,
    public readonly x33: number,
    public readonly x34: number,
    public readonly x41: number,
    public readonly x42: number,
    public readonly x43: number,
    public readonly x44: number,
  ) {}

  get repr() {
    return `(
  ${this.x11}, ${this.x12}, ${this.x13}, ${this.x14},
  ${this.x21}, ${this.x22}, ${this.x23}, ${this.x24},
  ${this.x31}, ${this.x32}, ${this.x33}, ${this.x34},
  ${this.x41}, ${this.x42}, ${this.x43}, ${this.x44}
)`;
  }

  mul(other: Matrix4x4): Matrix4x4 {
    return new Matrix4x4(
      this.x11 * other.x11 +
        this.x12 * other.x21 +
        this.x13 * other.x31 +
        this.x14 * other.x41,
      this.x11 * other.x12 +
        this.x12 * other.x22 +
        this.x13 * other.x32 +
        this.x14 * other.x42,
      this.x11 * other.x13 +
        this.x12 * other.x23 +
        this.x13 * other.x33 +
        this.x14 * other.x43,
      this.x11 * other.x14 +
        this.x12 * other.x24 +
        this.x13 * other.x34 +
        this.x14 * other.x44,

      this.x21 * other.x11 +
        this.x22 * other.x21 +
        this.x23 * other.x31 +
        this.x24 * other.x41,
      this.x21 * other.x12 +
        this.x22 * other.x22 +
        this.x23 * other.x32 +
        this.x24 * other.x42,
      this.x21 * other.x13 +
        this.x22 * other.x23 +
        this.x23 * other.x33 +
        this.x24 * other.x43,
      this.x21 * other.x14 +
        this.x22 * other.x24 +
        this.x23 * other.x34 +
        this.x24 * other.x44,

      this.x31 * other.x11 +
        this.x32 * other.x21 +
        this.x33 * other.x31 +
        this.x34 * other.x41,
      this.x31 * other.x12 +
        this.x32 * other.x22 +
        this.x33 * other.x32 +
        this.x34 * other.x42,
      this.x31 * other.x13 +
        this.x32 * other.x23 +
        this.x33 * other.x33 +
        this.x34 * other.x43,
      this.x31 * other.x14 +
        this.x32 * other.x24 +
        this.x33 * other.x34 +
        this.x34 * other.x44,

      this.x41 * other.x11 +
        this.x42 * other.x21 +
        this.x43 * other.x31 +
        this.x44 * other.x41,
      this.x41 * other.x12 +
        this.x42 * other.x22 +
        this.x43 * other.x32 +
        this.x44 * other.x42,
      this.x41 * other.x13 +
        this.x42 * other.x23 +
        this.x43 * other.x33 +
        this.x44 * other.x43,
      this.x41 * other.x14 +
        this.x42 * other.x24 +
        this.x43 * other.x34 +
        this.x44 * other.x44,
    );
  }

  mulMat4x3(other: Matrix4x3): Matrix4x3 {
    return new Matrix4x3(
      this.x11 * other.x11 +
        this.x12 * other.x21 +
        this.x13 * other.x31 +
        this.x14 * other.x41,
      this.x11 * other.x12 +
        this.x12 * other.x22 +
        this.x13 * other.x32 +
        this.x14 * other.x42,
      this.x11 * other.x13 +
        this.x12 * other.x23 +
        this.x13 * other.x33 +
        this.x14 * other.x43,

      this.x21 * other.x11 +
        this.x22 * other.x21 +
        this.x23 * other.x31 +
        this.x24 * other.x41,
      this.x21 * other.x12 +
        this.x22 * other.x22 +
        this.x23 * other.x32 +
        this.x24 * other.x42,
      this.x21 * other.x13 +
        this.x22 * other.x23 +
        this.x23 * other.x33 +
        this.x24 * other.x43,

      this.x31 * other.x11 +
        this.x32 * other.x21 +
        this.x33 * other.x31 +
        this.x34 * other.x41,
      this.x31 * other.x12 +
        this.x32 * other.x22 +
        this.x33 * other.x32 +
        this.x34 * other.x42,
      this.x31 * other.x13 +
        this.x32 * other.x23 +
        this.x33 * other.x33 +
        this.x34 * other.x43,

      this.x41 * other.x11 +
        this.x42 * other.x21 +
        this.x43 * other.x31 +
        this.x44 * other.x41,
      this.x41 * other.x12 +
        this.x42 * other.x22 +
        this.x43 * other.x32 +
        this.x44 * other.x42,
      this.x41 * other.x13 +
        this.x42 * other.x23 +
        this.x43 * other.x33 +
        this.x44 * other.x43,
    );
  }

  add(other: Matrix4x4): Matrix4x4 {
    return new Matrix4x4(
      this.x11 + other.x11,
      this.x12 + other.x12,
      this.x13 + other.x13,
      this.x14 + other.x14,
      this.x21 + other.x21,
      this.x22 + other.x22,
      this.x23 + other.x23,
      this.x24 + other.x24,
      this.x31 + other.x31,
      this.x32 + other.x32,
      this.x33 + other.x33,
      this.x34 + other.x34,
      this.x41 + other.x41,
      this.x42 + other.x42,
      this.x43 + other.x43,
      this.x44 + other.x44,
    );
  }

  sub(other: Matrix4x4): Matrix4x4 {
    return new Matrix4x4(
      this.x11 - other.x11,
      this.x12 - other.x12,
      this.x13 - other.x13,
      this.x14 - other.x14,
      this.x21 - other.x21,
      this.x22 - other.x22,
      this.x23 - other.x23,
      this.x24 - other.x24,
      this.x31 - other.x31,
      this.x32 - other.x32,
      this.x33 - other.x33,
      this.x34 - other.x34,
      this.x41 - other.x41,
      this.x42 - other.x42,
      this.x43 - other.x43,
      this.x44 - other.x44,
    );
  }

  scale(scalar: number): Matrix4x4 {
    return new Matrix4x4(
      this.x11 * scalar,
      this.x12 * scalar,
      this.x13 * scalar,
      this.x14 * scalar,
      this.x21 * scalar,
      this.x22 * scalar,
      this.x23 * scalar,
      this.x24 * scalar,
      this.x31 * scalar,
      this.x32 * scalar,
      this.x33 * scalar,
      this.x34 * scalar,
      this.x41 * scalar,
      this.x42 * scalar,
      this.x43 * scalar,
      this.x44 * scalar,
    );
  }

  product(v: ColVec4): ColVec4 {
    return new ColVec4(
      this.x11 * v.x1 + this.x12 * v.x2 + this.x13 * v.x3 + this.x14 * v.x4,
      this.x21 * v.x1 + this.x22 * v.x2 + this.x23 * v.x3 + this.x24 * v.x4,
      this.x31 * v.x1 + this.x32 * v.x2 + this.x33 * v.x3 + this.x34 * v.x4,
      this.x41 * v.x1 + this.x42 * v.x2 + this.x43 * v.x3 + this.x44 * v.x4,
    );
  }

  det(): number {
    // Calculate the determinant using cofactor expansion along the first row
    return (
      this.x11 * this._cofactor(0, 0) -
      this.x12 * this._cofactor(0, 1) +
      this.x13 * this._cofactor(0, 2) -
      this.x14 * this._cofactor(0, 3)
    );
  }

  // Helper method to calculate the cofactor of element at (row, col)
  private _cofactor(row: number, col: number): number {
    const minor = this._minor(row, col);
    // Determine sign based on position: (-1)^(row+col)
    const sign = (row + col) % 2 === 0 ? 1 : -1;
    return sign * minor;
  }

  // Helper method to calculate the minor of element at (row, col)
  private _minor(row: number, col: number): number {
    // Create a 3x3 submatrix by removing the row and column
    const elements: number[] = [];
    for (let i = 0; i < 4; i++) {
      if (i === row) continue;
      for (let j = 0; j < 4; j++) {
        if (j === col) continue;
        // Get the element from the original matrix
        elements.push(this._getElement(i, j));
      }
    }

    // Calculate determinant of the 3x3 submatrix
    return (
      elements[0] * (elements[4] * elements[8] - elements[5] * elements[7]) -
      elements[1] * (elements[3] * elements[8] - elements[5] * elements[6]) +
      elements[2] * (elements[3] * elements[7] - elements[4] * elements[6])
    );
  }

  // Helper method to get element at position (i, j)
  private _getElement(i: number, j: number): number {
    if (i === 0) {
      return j === 0
        ? this.x11
        : j === 1
          ? this.x12
          : j === 2
            ? this.x13
            : this.x14;
    } else if (i === 1) {
      return j === 0
        ? this.x21
        : j === 1
          ? this.x22
          : j === 2
            ? this.x23
            : this.x24;
    } else if (i === 2) {
      return j === 0
        ? this.x31
        : j === 1
          ? this.x32
          : j === 2
            ? this.x33
            : this.x34;
    } else {
      return j === 0
        ? this.x41
        : j === 1
          ? this.x42
          : j === 2
            ? this.x43
            : this.x44;
    }
  }

  transpose(): Matrix4x4 {
    return new Matrix4x4(
      this.x11,
      this.x21,
      this.x31,
      this.x41,
      this.x12,
      this.x22,
      this.x32,
      this.x42,
      this.x13,
      this.x23,
      this.x33,
      this.x43,
      this.x14,
      this.x24,
      this.x34,
      this.x44,
    );
  }

  inverse(): Matrix4x4 {
    const det = this.det();
    if (det === 0) {
      throw new Error("Matrix is singular and cannot be inverted");
    }

    // Calculate the adjugate matrix (transpose of cofactor matrix)
    const adjugate = new Matrix4x4(
      this._cofactor(0, 0),
      this._cofactor(1, 0),
      this._cofactor(2, 0),
      this._cofactor(3, 0),
      this._cofactor(0, 1),
      this._cofactor(1, 1),
      this._cofactor(2, 1),
      this._cofactor(3, 1),
      this._cofactor(0, 2),
      this._cofactor(1, 2),
      this._cofactor(2, 2),
      this._cofactor(3, 2),
      this._cofactor(0, 3),
      this._cofactor(1, 3),
      this._cofactor(2, 3),
      this._cofactor(3, 3),
    );

    // Divide adjugate by determinant
    return adjugate.scale(1 / det);
  }
}

export const diagonalMatrix4x4 = (
  x11: number,
  x22: number,
  x33: number,
  x44: number,
): Matrix4x4 => {
  return new Matrix4x4(x11, 0, 0, 0, 0, x22, 0, 0, 0, 0, x33, 0, 0, 0, 0, x44);
};
export const IDENTITY_MATRIX_4x4 = diagonalMatrix4x4(1, 1, 1, 1);

export class Matrix3x4 {
  constructor(
    public readonly x11: number,
    public readonly x12: number,
    public readonly x13: number,
    public readonly x14: number,
    public readonly x21: number,
    public readonly x22: number,
    public readonly x23: number,
    public readonly x24: number,
    public readonly x31: number,
    public readonly x32: number,
    public readonly x33: number,
    public readonly x34: number,
  ) {}

  get repr() {
    return `(
  ${this.x11}, ${this.x12}, ${this.x13}, ${this.x14},
  ${this.x21}, ${this.x22}, ${this.x23}, ${this.x24},
  ${this.x31}, ${this.x32}, ${this.x33}, ${this.x34}
)`;
  }

  add(other: Matrix3x4): Matrix3x4 {
    return new Matrix3x4(
      this.x11 + other.x11,
      this.x12 + other.x12,
      this.x13 + other.x13,
      this.x14 + other.x14,
      this.x21 + other.x21,
      this.x22 + other.x22,
      this.x23 + other.x23,
      this.x24 + other.x24,
      this.x31 + other.x31,
      this.x32 + other.x32,
      this.x33 + other.x33,
      this.x34 + other.x34,
    );
  }

  sub(other: Matrix3x4): Matrix3x4 {
    return new Matrix3x4(
      this.x11 - other.x11,
      this.x12 - other.x12,
      this.x13 - other.x13,
      this.x14 - other.x14,
      this.x21 - other.x21,
      this.x22 - other.x22,
      this.x23 - other.x23,
      this.x24 - other.x24,
      this.x31 - other.x31,
      this.x32 - other.x32,
      this.x33 - other.x33,
      this.x34 - other.x34,
    );
  }

  scale(scalar: number): Matrix3x4 {
    return new Matrix3x4(
      this.x11 * scalar,
      this.x12 * scalar,
      this.x13 * scalar,
      this.x14 * scalar,
      this.x21 * scalar,
      this.x22 * scalar,
      this.x23 * scalar,
      this.x24 * scalar,
      this.x31 * scalar,
      this.x32 * scalar,
      this.x33 * scalar,
      this.x34 * scalar,
    );
  }

  mul(other: Matrix4x4): Matrix3x4 {
    return new Matrix3x4(
      this.x11 * other.x11 +
        this.x12 * other.x21 +
        this.x13 * other.x31 +
        this.x14 * other.x41,
      this.x11 * other.x12 +
        this.x12 * other.x22 +
        this.x13 * other.x32 +
        this.x14 * other.x42,
      this.x11 * other.x13 +
        this.x12 * other.x23 +
        this.x13 * other.x33 +
        this.x14 * other.x43,
      this.x11 * other.x14 +
        this.x12 * other.x24 +
        this.x13 * other.x34 +
        this.x14 * other.x44,

      this.x21 * other.x11 +
        this.x22 * other.x21 +
        this.x23 * other.x31 +
        this.x24 * other.x41,
      this.x21 * other.x12 +
        this.x22 * other.x22 +
        this.x23 * other.x32 +
        this.x24 * other.x42,
      this.x21 * other.x13 +
        this.x22 * other.x23 +
        this.x23 * other.x33 +
        this.x24 * other.x43,
      this.x21 * other.x14 +
        this.x22 * other.x24 +
        this.x23 * other.x34 +
        this.x24 * other.x44,

      this.x31 * other.x11 +
        this.x32 * other.x21 +
        this.x33 * other.x31 +
        this.x34 * other.x41,
      this.x31 * other.x12 +
        this.x32 * other.x22 +
        this.x33 * other.x32 +
        this.x34 * other.x42,
      this.x31 * other.x13 +
        this.x32 * other.x23 +
        this.x33 * other.x33 +
        this.x34 * other.x43,
      this.x31 * other.x14 +
        this.x32 * other.x24 +
        this.x33 * other.x34 +
        this.x34 * other.x44,
    );
  }

  mulMat4x3(other: Matrix4x3): Matrix3x3 {
    return new Matrix3x3(
      this.x11 * other.x11 +
        this.x12 * other.x21 +
        this.x13 * other.x31 +
        this.x14 * other.x41,
      this.x11 * other.x12 +
        this.x12 * other.x22 +
        this.x13 * other.x32 +
        this.x14 * other.x42,
      this.x11 * other.x13 +
        this.x12 * other.x23 +
        this.x13 * other.x33 +
        this.x14 * other.x43,

      this.x21 * other.x11 +
        this.x22 * other.x21 +
        this.x23 * other.x31 +
        this.x24 * other.x41,
      this.x21 * other.x12 +
        this.x22 * other.x22 +
        this.x23 * other.x32 +
        this.x24 * other.x42,
      this.x21 * other.x13 +
        this.x22 * other.x23 +
        this.x23 * other.x33 +
        this.x24 * other.x43,

      this.x31 * other.x11 +
        this.x32 * other.x21 +
        this.x33 * other.x31 +
        this.x34 * other.x41,
      this.x31 * other.x12 +
        this.x32 * other.x22 +
        this.x33 * other.x32 +
        this.x34 * other.x42,
      this.x31 * other.x13 +
        this.x32 * other.x23 +
        this.x33 * other.x33 +
        this.x34 * other.x43,
    );
  }

  product(v: ColVec4): ColVec3 {
    return new ColVec3(
      this.x11 * v.x1 + this.x12 * v.x2 + this.x13 * v.x3 + this.x14 * v.x4,
      this.x21 * v.x1 + this.x22 * v.x2 + this.x23 * v.x3 + this.x24 * v.x4,
      this.x31 * v.x1 + this.x32 * v.x2 + this.x33 * v.x3 + this.x34 * v.x4,
    );
  }

  transpose(): Matrix4x3 {
    return new Matrix4x3(
      this.x11,
      this.x21,
      this.x31,
      this.x12,
      this.x22,
      this.x32,
      this.x13,
      this.x23,
      this.x33,
      this.x14,
      this.x24,
      this.x34,
    );
  }
}

export class Matrix4x3 {
  constructor(
    public readonly x11: number,
    public readonly x12: number,
    public readonly x13: number,
    public readonly x21: number,
    public readonly x22: number,
    public readonly x23: number,
    public readonly x31: number,
    public readonly x32: number,
    public readonly x33: number,
    public readonly x41: number,
    public readonly x42: number,
    public readonly x43: number,
  ) {}

  get repr() {
    return `(
  ${this.x11}, ${this.x12}, ${this.x13},
  ${this.x21}, ${this.x22}, ${this.x23},
  ${this.x31}, ${this.x32}, ${this.x33},
  ${this.x41}, ${this.x42}, ${this.x43}
)`;
  }

  add(other: Matrix4x3): Matrix4x3 {
    return new Matrix4x3(
      this.x11 + other.x11,
      this.x12 + other.x12,
      this.x13 + other.x13,
      this.x21 + other.x21,
      this.x22 + other.x22,
      this.x23 + other.x23,
      this.x31 + other.x31,
      this.x32 + other.x32,
      this.x33 + other.x33,
      this.x41 + other.x41,
      this.x42 + other.x42,
      this.x43 + other.x43,
    );
  }

  sub(other: Matrix4x3): Matrix4x3 {
    return new Matrix4x3(
      this.x11 - other.x11,
      this.x12 - other.x12,
      this.x13 - other.x13,
      this.x21 - other.x21,
      this.x22 - other.x22,
      this.x23 - other.x23,
      this.x31 - other.x31,
      this.x32 - other.x32,
      this.x33 - other.x33,
      this.x41 - other.x41,
      this.x42 - other.x42,
      this.x43 - other.x43,
    );
  }

  scale(scalar: number): Matrix4x3 {
    return new Matrix4x3(
      this.x11 * scalar,
      this.x12 * scalar,
      this.x13 * scalar,
      this.x21 * scalar,
      this.x22 * scalar,
      this.x23 * scalar,
      this.x31 * scalar,
      this.x32 * scalar,
      this.x33 * scalar,
      this.x41 * scalar,
      this.x42 * scalar,
      this.x43 * scalar,
    );
  }

  mul(other: Matrix3x3): Matrix4x3 {
    return new Matrix4x3(
      this.x11 * other.x11 + this.x12 * other.x21 + this.x13 * other.x31,
      this.x11 * other.x12 + this.x12 * other.x22 + this.x13 * other.x32,
      this.x11 * other.x13 + this.x12 * other.x23 + this.x13 * other.x33,

      this.x21 * other.x11 + this.x22 * other.x21 + this.x23 * other.x31,
      this.x21 * other.x12 + this.x22 * other.x22 + this.x23 * other.x32,
      this.x21 * other.x13 + this.x22 * other.x23 + this.x23 * other.x33,

      this.x31 * other.x11 + this.x32 * other.x21 + this.x33 * other.x31,
      this.x31 * other.x12 + this.x32 * other.x22 + this.x33 * other.x32,
      this.x31 * other.x13 + this.x32 * other.x23 + this.x33 * other.x33,

      this.x41 * other.x11 + this.x42 * other.x21 + this.x43 * other.x31,
      this.x41 * other.x12 + this.x42 * other.x22 + this.x43 * other.x32,
      this.x41 * other.x13 + this.x42 * other.x23 + this.x43 * other.x33,
    );
  }

  mulMat3x4(other: Matrix3x4): Matrix4x4 {
    return new Matrix4x4(
      this.x11 * other.x11 + this.x12 * other.x21 + this.x13 * other.x31,
      this.x11 * other.x12 + this.x12 * other.x22 + this.x13 * other.x32,
      this.x11 * other.x13 + this.x12 * other.x23 + this.x13 * other.x33,
      this.x11 * other.x14 + this.x12 * other.x24 + this.x13 * other.x34,

      this.x21 * other.x11 + this.x22 * other.x21 + this.x23 * other.x31,
      this.x21 * other.x12 + this.x22 * other.x22 + this.x23 * other.x32,
      this.x21 * other.x13 + this.x22 * other.x23 + this.x23 * other.x33,
      this.x21 * other.x14 + this.x22 * other.x24 + this.x23 * other.x34,

      this.x31 * other.x11 + this.x32 * other.x21 + this.x33 * other.x31,
      this.x31 * other.x12 + this.x32 * other.x22 + this.x33 * other.x32,
      this.x31 * other.x13 + this.x32 * other.x23 + this.x33 * other.x33,
      this.x31 * other.x14 + this.x32 * other.x24 + this.x33 * other.x34,

      this.x41 * other.x11 + this.x42 * other.x21 + this.x43 * other.x31,
      this.x41 * other.x12 + this.x42 * other.x22 + this.x43 * other.x32,
      this.x41 * other.x13 + this.x42 * other.x23 + this.x43 * other.x33,
      this.x41 * other.x14 + this.x42 * other.x24 + this.x43 * other.x34,
    );
  }

  product(v: ColVec3): ColVec4 {
    return new ColVec4(
      this.x11 * v.x1 + this.x12 * v.x2 + this.x13 * v.x3,
      this.x21 * v.x1 + this.x22 * v.x2 + this.x23 * v.x3,
      this.x31 * v.x1 + this.x32 * v.x2 + this.x33 * v.x3,
      this.x41 * v.x1 + this.x42 * v.x2 + this.x43 * v.x3,
    );
  }

  transpose(): Matrix3x4 {
    return new Matrix3x4(
      this.x11,
      this.x21,
      this.x31,
      this.x41,
      this.x12,
      this.x22,
      this.x32,
      this.x42,
      this.x13,
      this.x23,
      this.x33,
      this.x43,
    );
  }
}
