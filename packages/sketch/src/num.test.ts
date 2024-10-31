import { test } from "vitest";
import { asNum } from "./num";

import { ex } from "./test-utils";

test("literal nums", () => {
  ex(asNum(1)).toBe(1);
  ex(asNum(2)).toBe(2);
  ex(asNum(0)).toBe(0);
  ex(asNum(-1)).toBe(-1);
});

test("num addition", () => {
  ex(asNum(1).add(asNum(2))).toBeCloseTo(3);
  ex(asNum(1).add(asNum(0))).toBeCloseTo(1);
  ex(asNum(1).add(asNum(-1))).toBeCloseTo(0);
  ex(asNum(0.3).add(asNum(0.1))).toBeCloseTo(0.4);
});

test("num subtraction", () => {
  ex(asNum(1).sub(asNum(2))).toBeCloseTo(-1);
  ex(asNum(1).sub(asNum(0))).toBeCloseTo(1);
  ex(asNum(1).sub(asNum(-1))).toBeCloseTo(2);
  ex(asNum(0.3).sub(asNum(0.1))).toBeCloseTo(0.2);
});

test("num multiplication", () => {
  ex(asNum(1).mul(asNum(2))).toBeCloseTo(2);
  ex(asNum(1).mul(asNum(0))).toBeCloseTo(0);
  ex(asNum(1).mul(asNum(-1))).toBeCloseTo(-1);
  ex(asNum(0.3).mul(asNum(0.1))).toBeCloseTo(0.03);
});

test("num division", () => {
  ex(asNum(1).div(asNum(2))).toBeCloseTo(0.5);
  ex(asNum(1).div(asNum(1))).toBeCloseTo(1);
  ex(asNum(1).div(asNum(-1))).toBeCloseTo(-1);
  ex(asNum(0.3).div(asNum(0.1))).toBeCloseTo(3);
});

test("num square root", () => {
  ex(asNum(1).sqrt()).toBeCloseTo(1);
  ex(asNum(4).sqrt()).toBeCloseTo(2);
  ex(asNum(0.25).sqrt()).toBeCloseTo(0.5);
});

test("num negation", () => {
  ex(asNum(1).neg()).toBeCloseTo(-1);
  ex(asNum(-1.2).neg()).toBeCloseTo(1.2);
  ex(asNum(0).neg()).toBeCloseTo(0);
});

test("num sign", () => {
  ex(asNum(10).sign()).toBeCloseTo(1);
  ex(asNum(-1.2).sign()).toBeCloseTo(-1);
  ex(asNum(0).sign()).toBeCloseTo(0);
});

test("num mod", () => {
  ex(asNum(10).mod(asNum(3))).toBeCloseTo(1);
  ex(asNum(10).mod(asNum(5))).toBeCloseTo(0);
  ex(asNum(10).mod(asNum(7))).toBeCloseTo(3);
});

test("num cos", () => {
  ex(asNum(0).cos()).toBeCloseTo(1);
  ex(asNum(Math.PI).cos()).toBeCloseTo(-1);
  ex(asNum(Math.PI / 2).cos()).toBeCloseTo(0);
});

test("num acos", () => {
  ex(asNum(1).acos()).toBeCloseTo(0);
  ex(asNum(-1).acos()).toBeCloseTo(Math.PI);
  ex(asNum(0).acos()).toBeCloseTo(Math.PI / 2);
});

test("num sin", () => {
  ex(asNum(0).sin()).toBeCloseTo(0);
  ex(asNum(Math.PI).sin()).toBeCloseTo(0);
  ex(asNum(Math.PI / 2).sin()).toBeCloseTo(1);
});

test("num asin", () => {
  ex(asNum(0).asin()).toBeCloseTo(0);
  ex(asNum(1).asin()).toBeCloseTo(Math.PI / 2);
  ex(asNum(-1).asin()).toBeCloseTo(-Math.PI / 2);
});

test("num tan", () => {
  ex(asNum(0).tan()).toBeCloseTo(0);
  ex(asNum(Math.PI).tan()).toBeCloseTo(0);
  ex(asNum(Math.PI / 4).tan()).toBeCloseTo(1);
});

test("num atan", () => {
  ex(asNum(0).atan()).toBeCloseTo(0);
  ex(asNum(1).atan()).toBeCloseTo(Math.PI / 4);
  ex(asNum(-1).atan()).toBeCloseTo(-Math.PI / 4);
});

test("num log", () => {
  ex(asNum(1).log()).toBeCloseTo(0);
  ex(asNum(Math.E).log()).toBeCloseTo(1);
  ex(asNum(10).log()).toBeCloseTo(Math.log(10));
});

test("num exp", () => {
  ex(asNum(0).exp()).toBeCloseTo(1);
  ex(asNum(1).exp()).toBeCloseTo(Math.E);
  ex(asNum(2).exp()).toBeCloseTo(Math.E ** 2);
});

test("num square", () => {
  ex(asNum(0).square()).toBeCloseTo(0);
  ex(asNum(1.5).square()).toBeCloseTo(2.25);
  ex(asNum(2).square()).toBeCloseTo(4);
});

test("num abs", () => {
  ex(asNum(0).abs()).toBeCloseTo(0);
  ex(asNum(1.5).abs()).toBeCloseTo(1.5);
  ex(asNum(-2).abs()).toBeCloseTo(2);
});

test("num compare", () => {
  ex(asNum(0).compare(0)).toBeCloseTo(0);
  ex(asNum(1).compare(0)).toBeCloseTo(1);
  ex(asNum(0).compare(1)).toBeCloseTo(-1);
});

test("num and", () => {
  ex(asNum(0).and(0)).toBeCloseTo(0);
  ex(asNum(1).and(0)).toBeCloseTo(0);
  ex(asNum(0).and(1)).toBeCloseTo(0);
  ex(asNum(1).and(1)).toBeCloseTo(1);
  ex(asNum(1).and(-1)).toBeCloseTo(-1);
});

test("num or", () => {
  ex(asNum(0).or(0)).toBeCloseTo(0);
  ex(asNum(1).or(0)).toBeCloseTo(1);
  ex(asNum(0).or(1)).toBeCloseTo(1);
  ex(asNum(1).or(1)).toBeCloseTo(1);
  ex(asNum(1).or(-1)).toBeCloseTo(1);
  ex(asNum(-1).or(1)).toBeCloseTo(-1);
});

test("num max", () => {
  ex(asNum(0).max(0)).toBeCloseTo(0);
  ex(asNum(1).max(0)).toBeCloseTo(1);
  ex(asNum(0).max(1)).toBeCloseTo(1);
  ex(asNum(1).max(1)).toBeCloseTo(1);
  ex(asNum(1).max(-1)).toBeCloseTo(1);
  ex(asNum(-1).max(1)).toBeCloseTo(1);
});

test("num min", () => {
  ex(asNum(0).min(0)).toBeCloseTo(0);
  ex(asNum(1).min(0)).toBeCloseTo(0);
  ex(asNum(0).min(1)).toBeCloseTo(0);
  ex(asNum(1).min(1)).toBeCloseTo(1);
  ex(asNum(1).min(-1)).toBeCloseTo(-1);
  ex(asNum(-1).min(1)).toBeCloseTo(-1);
});

test("num less than", () => {
  ex(asNum(0).lessThan(0)).toBeCloseTo(0);
  ex(asNum(1).lessThan(0)).toBeCloseTo(0);
  ex(asNum(0).lessThan(1)).toBeCloseTo(1);
  ex(asNum(1).lessThan(1)).toBeCloseTo(0);
  ex(asNum(1).lessThan(-1)).toBeCloseTo(0);
  ex(asNum(-1).lessThan(1)).toBeCloseTo(1);
});

test("num less than or equal", () => {
  ex(asNum(0).lessThanOrEqual(0)).toBeCloseTo(1);
  ex(asNum(1).lessThanOrEqual(0)).toBeCloseTo(0);
  ex(asNum(0).lessThanOrEqual(1)).toBeCloseTo(1);
  ex(asNum(1).lessThanOrEqual(1)).toBeCloseTo(1);
  ex(asNum(1).lessThanOrEqual(-1)).toBeCloseTo(0);
  ex(asNum(-1).lessThanOrEqual(1)).toBeCloseTo(1);
});

test("num greater than", () => {
  ex(asNum(0).greaterThan(0)).toBeCloseTo(0);
  ex(asNum(1).greaterThan(0)).toBeCloseTo(1);
  ex(asNum(0).greaterThan(1)).toBeCloseTo(0);
  ex(asNum(1).greaterThan(1)).toBeCloseTo(0);
  ex(asNum(1).greaterThan(-1)).toBeCloseTo(1);
  ex(asNum(-1).greaterThan(1)).toBeCloseTo(0);
});

test("num greater than or equal", () => {
  ex(asNum(0).greaterThanOrEqual(0)).toBeCloseTo(1);
  ex(asNum(1).greaterThanOrEqual(0)).toBeCloseTo(1);
  ex(asNum(0).greaterThanOrEqual(1)).toBeCloseTo(0);
  ex(asNum(1).greaterThanOrEqual(1)).toBeCloseTo(1);
  ex(asNum(1).greaterThanOrEqual(-1)).toBeCloseTo(1);
  ex(asNum(-1).greaterThanOrEqual(1)).toBeCloseTo(0);
});
test("num equals", () => {
  ex(asNum(0).equals(0)).toBeCloseTo(1);
  ex(asNum(1).equals(0)).toBeCloseTo(0);
  ex(asNum(0).equals(1)).toBeCloseTo(0);
  ex(asNum(1).equals(1)).toBeCloseTo(1);
  ex(asNum(1).equals(-1)).toBeCloseTo(0);
  ex(asNum(-1).equals(1)).toBeCloseTo(0);
});
