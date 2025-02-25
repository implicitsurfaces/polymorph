import { describe, test, expect } from "vitest";

import { asNum } from "../num";
import {
  ColVec3,
  IDENTITY_MATRIX_3x3,
  Matrix3x3,
  RowVec3,
  Matrix2x2,
  ColVec2,
  RowVec2,
  IDENTITY_MATRIX_2x2,
} from "./matrices";
import { evaluate } from "../utils/evaluate";

const c = (numbers: number[]) => {
  return numbers.map((n) => expect.closeTo(n, 5));
};

describe("Matrix2x2", () => {
  const m2 = (numbers: number[]) => {
    // @ts-expect-error - I actually pass a tuple of 4 numbers
    return new Matrix2x2(...numbers.map(asNum));
  };

  const column = (a: number, b: number): ColVec2 => {
    return new ColVec2(asNum(a), asNum(b));
  };

  const row = (a: number, b: number): RowVec2 => {
    return new RowVec2(asNum(a), asNum(b));
  };

  test("matrix multiplication by identity", () => {
    const a = m2([1, 2, 3, 4]);
    const b = IDENTITY_MATRIX_2x2;

    const result = a.mul(b);

    expect(evaluate(result)).toEqual(c([1, 2, 3, 4]));
  });

  test("matrix multiplication", () => {
    const a = m2([1, 2, 3, 4]);
    const b = m2([1, 3, 2, 4]);

    const result = a.mul(b);

    expect(evaluate(result)).toEqual(c([5, 11, 11, 25]));
  });

  test("transpose", () => {
    const a = m2([1, 2, 3, 4]);

    const result = a.transpose();

    expect(evaluate(result)).toEqual(c([1, 3, 2, 4]));
  });

  test("inverse", () => {
    const a = m2([4, 3, 2, 1]);

    const result = a.inverse();
    const shouldBeIdentity = a.mul(result);

    expect(evaluate(shouldBeIdentity)).toEqual(c([1, 0, 0, 1]));
  });

  test("determinant", () => {
    const a = m2([4, 3, 2, 1]);

    const result = a.det();

    expect(evaluate(result)).toBeCloseTo(-2);
  });

  test("product on an column vector", () => {
    const a = m2([1, 2, 3, 4]);
    const b = column(1, 2);

    const result = a.product(b);

    expect(evaluate(result)).toEqual(c([5, 11]));
  });

  test("product on an row vector", () => {
    const a = m2([1, 2, 3, 4]);
    const b = row(1, 2);

    const result = b.product(a);

    expect(evaluate(result)).toEqual(c([7, 10]));
  });

  test("dot product", () => {
    const a = row(1, 2);
    const b = column(1, 2);

    const result = a.dot(b);

    expect(evaluate(result)).toBeCloseTo(5);
  });

  test("trace", () => {
    const a = m2([1, 2, 3, 4]);

    const result = a.trace();

    expect(evaluate(result)).toBeCloseTo(5);
  });

  test("eigenvalues", () => {
    const a = m2([2, 1, 0, 3]);
    const result = a.eigenvalues();
    expect(evaluate(result)).toEqual(c([2, 3]));

    const b = m2([1, 2, 2, 1]);
    const result2 = b.eigenvalues();
    expect(evaluate(result2)).toEqual(c([-1, 3]));

    const d = m2([5, 4, 1, 2]);
    const result3 = d.eigenvalues();
    expect(evaluate(result3)).toEqual(c([1, 6]));
  });

  test("eigenvectors", () => {
    const a = m2([2, 1, 0, 3]);
    const result = a.eigenvectors();

    const mat = new Matrix2x2(
      result[0].x1,
      result[1].x1,
      result[0].x2,
      result[1].x2,
    );
    const shouldBeDiagonal = mat.inverse().mul(a).mul(mat);
    expect(evaluate(shouldBeDiagonal)).toEqual(c([2, 0, 0, 3]));

    const d = m2([5, 4, 1, 2]);
    const result3 = d.eigenvectors();
    const mat3 = new Matrix2x2(
      result3[0].x1,
      result3[1].x1,
      result3[0].x2,
      result3[1].x2,
    );
    const shouldBeDiagonal3 = mat3.inverse().mul(d).mul(mat3);
    expect(evaluate(shouldBeDiagonal3)).toEqual(c([1, 0, 0, 6]));

    const diag = m2([10, 0, 0, 3]);
    const result4 = diag.eigenvectors();
    const mat4 = new Matrix2x2(
      result4[0].x1,
      result4[1].x1,
      result4[0].x2,
      result4[1].x2,
    );
    const shouldBeDiagonal4 = mat4.inverse().mul(diag).mul(mat4);
    expect(evaluate(shouldBeDiagonal4)).toEqual(c([3, 0, 0, 10]));
  });
});

describe("Matrix3x3", () => {
  const m3 = (numbers: number[]) => {
    // @ts-expect-error - I actually pass a tuple of 9 numbers
    return new Matrix3x3(...numbers.map(asNum));
  };

  const column = (a: number, b: number, c: number): ColVec3 => {
    return new ColVec3(asNum(a), asNum(b), asNum(c));
  };

  const row = (a: number, b: number, c: number): RowVec3 => {
    return new RowVec3(asNum(a), asNum(b), asNum(c));
  };

  test("matrix multiplication by identity", () => {
    const a = m3([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const b = IDENTITY_MATRIX_3x3;

    const result = a.mul(b);

    expect(evaluate(result)).toEqual(c([1, 2, 3, 4, 5, 6, 7, 8, 9]));
  });

  test("matrix multiplication", () => {
    const a = m3([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const b = m3([1, 4, 7, 2, 5, 8, 3, 6, 9]);

    const result = a.mul(b);

    expect(evaluate(result)).toEqual(
      c([14, 32, 50, 32, 77, 122, 50, 122, 194]),
    );
  });

  test("transpose", () => {
    const a = m3([1, 2, 3, 4, 5, 6, 7, 8, 9]);

    const result = a.transpose();

    expect(evaluate(result)).toEqual(c([1, 4, 7, 2, 5, 8, 3, 6, 9]));
  });

  test("inverse", () => {
    const a = m3([2, 1, 1, 1, 3, 2, 1, 2, 4]);

    const result = a.inverse();
    const shouldBeIdentity = a.mul(result);

    expect(evaluate(shouldBeIdentity)).toEqual(c([1, 0, 0, 0, 1, 0, 0, 0, 1]));
  });

  test("determinant", () => {
    const a = m3([4, 2, 0, 0, 3, 2, 0, 0, 1]);

    const result = a.det();

    expect(evaluate(result)).toBeCloseTo(12);
  });

  test("product on an column vector", () => {
    const a = m3([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const b = column(1, 2, 3);

    const result = a.product(b);

    expect(evaluate(result)).toEqual(c([14, 32, 50]));
  });

  test("product on an row vector", () => {
    const a = m3([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const b = row(1, 2, 3);

    const result = b.product(a);

    expect(evaluate(result)).toEqual(c([30, 36, 42]));
  });

  test("dot product", () => {
    const a = row(1, 2, 3);
    const b = column(1, 2, 3);

    const result = a.dot(b);

    expect(evaluate(result)).toBeCloseTo(14);
  });

  test("trace", () => {
    const a = m3([1, 2, 3, 4, 5, 6, 7, 8, 9]);

    const result = a.trace();

    expect(evaluate(result)).toBeCloseTo(15);
  });
});
