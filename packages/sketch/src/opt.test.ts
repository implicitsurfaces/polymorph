import { expect, test } from "vitest";

import { gradientDescentOpt, Gradient } from "./opt";
import { asNum, variable } from "./num";

test("Gradient of a single var", () => {
  const num = variable("x").square().add(1);

  const grad = new Gradient(num);

  expect(grad.at(new Map([["x", 0]])).get("x")).toBe(0);
  expect(grad.at(new Map([["x", 1]])).get("x")).toBe(2);
  expect(grad.at(new Map([["x", 2]])).get("x")).toBe(4);
  expect(grad.at(new Map([["x", 3]])).get("x")).toBe(6);
});

test("Gradient of multiple vars", () => {
  const num = variable("x").square().add(variable("y").square());

  const grad = new Gradient(num);
  const p = (x: number, y: number) =>
    new Map([
      ["x", x],
      ["y", y],
    ]);

  expect(grad.at(p(0, 0)).get("x")).toBe(0);
  expect(grad.at(p(0, 0)).get("y")).toBe(0);

  expect(grad.at(p(1, 0)).get("x")).toBe(2);
  expect(grad.at(p(1, 0)).get("y")).toBe(0);

  expect(grad.at(p(0, 1)).get("x")).toBe(0);
  expect(grad.at(p(0, 1)).get("y")).toBe(2);

  expect(grad.at(p(1, 1)).get("x")).toBe(2);

  expect(grad.at(p(1, 1)).get("y")).toBe(2);
});

test("Gradient of the rosensbrock function", () => {
  const num = asNum(1)
    .sub(variable("x"))
    .square()
    .add(asNum(100).mul(variable("y").sub(variable("x").square()).square()));

  const grad = new Gradient(num);
  const p = (x: number, y: number) =>
    new Map([
      ["x", x],
      ["y", y],
    ]);

  expect(grad.at(p(0, 0)).get("x")).toBe(-2);
  expect(grad.at(p(0, 0)).get("y")).toBe(0);
  expect(grad.at(p(1, 1)).get("x")).toBe(0);
  expect(grad.at(p(1, 1)).get("y")).toBe(0);
});

test("Gradient descent on a simple function", () => {
  const num = variable("x").square().add(1);

  const initialX = new Map([["x", 10]]);
  const result = gradientDescentOpt(num, initialX);

  expect(result.solution.get("x")).toBeCloseTo(0, 4);
});

test("Gradient descent on a two variable function", () => {
  const num = variable("x").square().add(variable("y").square());

  const initialX = new Map([
    ["x", 10],
    ["y", 10],
  ]);
  const result = gradientDescentOpt(num, initialX);

  expect(result.solution.get("x")).toBeCloseTo(0, 4);
  expect(result.solution.get("y")).toBeCloseTo(0, 4);
});

test("Gradient descent on the rosenbrock function", () => {
  const num = asNum(1)
    .sub(variable("x"))
    .square()
    .add(asNum(100).mul(variable("y").sub(variable("x").square()).square()));

  const initialX = new Map([
    ["x", -1],
    ["y", -1],
  ]);
  const result = gradientDescentOpt(num, initialX, {
    maxSteps: 10000,
    learningRate: 0.0001,
    momentum: 0.9,
  });

  expect(result.solution.get("x")).toBeCloseTo(1, 2);
  expect(result.solution.get("y")).toBeCloseTo(1, 2);
});
