import { test, expect } from "vitest";

import { asNum } from "../num";
import { simpleEval } from "../num-tree";
import { ColVec3, IDENTITY_MATRIX_3x3, Matrix3x3, RowVec3 } from "./matrices";

function evalMatrix(matrix: Matrix3x3) {
  return [
    simpleEval(matrix.x11.n),
    simpleEval(matrix.x12.n),
    simpleEval(matrix.x13.n),
    simpleEval(matrix.x21.n),
    simpleEval(matrix.x22.n),
    simpleEval(matrix.x23.n),
    simpleEval(matrix.x31.n),
    simpleEval(matrix.x32.n),
    simpleEval(matrix.x33.n),
  ];
}

function evalVec(vec: ColVec3 | RowVec3) {
  return [simpleEval(vec.x1.n), simpleEval(vec.x2.n), simpleEval(vec.x3.n)];
}

const m = (numbers: number[]) => {
  // @ts-expect-error - I actually pass a tuple of 9 numbers
  return new Matrix3x3(...numbers.map(asNum));
};

const column = (a: number, b: number, c: number): ColVec3 => {
  return new ColVec3(asNum(a), asNum(b), asNum(c));
};

const row = (a: number, b: number, c: number): RowVec3 => {
  return new RowVec3(asNum(a), asNum(b), asNum(c));
};

const c = (numbers: number[]) => {
  return numbers.map((n) => expect.closeTo(n, 5));
};

test("matrix multiplication by identity", () => {
  const a = m([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const b = IDENTITY_MATRIX_3x3;

  const result = a.mul(b);

  expect(evalMatrix(result)).toEqual(c([1, 2, 3, 4, 5, 6, 7, 8, 9]));
});

test("matrix multiplication", () => {
  const a = m([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const b = m([1, 4, 7, 2, 5, 8, 3, 6, 9]);

  const result = a.mul(b);

  expect(evalMatrix(result)).toEqual(
    c([14, 32, 50, 32, 77, 122, 50, 122, 194]),
  );
});

test("transpose", () => {
  const a = m([1, 2, 3, 4, 5, 6, 7, 8, 9]);

  const result = a.transpose();

  expect(evalMatrix(result)).toEqual(c([1, 4, 7, 2, 5, 8, 3, 6, 9]));
});

test("inverse", () => {
  const a = m([2, 1, 1, 1, 3, 2, 1, 2, 4]);

  const result = a.inverse();
  const shouldBeIdentity = a.mul(result);

  expect(evalMatrix(shouldBeIdentity)).toEqual(c([1, 0, 0, 0, 1, 0, 0, 0, 1]));
});

test("determinant", () => {
  const a = m([4, 2, 0, 0, 3, 2, 0, 0, 1]);

  const result = a.det();

  expect(simpleEval(result.n)).toBeCloseTo(12);
});

test("product on an column vector", () => {
  const a = m([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const b = column(1, 2, 3);

  const result = a.product(b);

  expect(evalVec(result)).toEqual(c([14, 32, 50]));
});

test("product on an row vector", () => {
  const a = m([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const b = row(1, 2, 3);

  const result = b.product(a);

  expect(evalVec(result)).toEqual(c([30, 36, 42]));
});

test("dot product", () => {
  const a = row(1, 2, 3);
  const b = column(1, 2, 3);

  const result = a.dot(b);

  expect(simpleEval(result.n)).toBeCloseTo(14);
});
