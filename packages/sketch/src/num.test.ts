import { test } from "vitest";
import { as_num } from "./num";

import { ex } from "./test-utils";

test("literal nums", () => {
  ex(as_num(1)).toBe(1);
  ex(as_num(2)).toBe(2);
  ex(as_num(0)).toBe(0);
  ex(as_num(-1)).toBe(-1);
});

test("num addition", () => {
  ex(as_num(1).add(as_num(2))).toBeCloseTo(3);
  ex(as_num(1).add(as_num(0))).toBeCloseTo(1);
  ex(as_num(1).add(as_num(-1))).toBeCloseTo(0);
  ex(as_num(0.3).add(as_num(0.1))).toBeCloseTo(0.4);
});

test("num subtraction", () => {
  ex(as_num(1).sub(as_num(2))).toBeCloseTo(-1);
  ex(as_num(1).sub(as_num(0))).toBeCloseTo(1);
  ex(as_num(1).sub(as_num(-1))).toBeCloseTo(2);
  ex(as_num(0.3).sub(as_num(0.1))).toBeCloseTo(0.2);
});

test("num multiplication", () => {
  ex(as_num(1).mul(as_num(2))).toBeCloseTo(2);
  ex(as_num(1).mul(as_num(0))).toBeCloseTo(0);
  ex(as_num(1).mul(as_num(-1))).toBeCloseTo(-1);
  ex(as_num(0.3).mul(as_num(0.1))).toBeCloseTo(0.03);
});

test("num division", () => {
  ex(as_num(1).div(as_num(2))).toBeCloseTo(0.5);
  ex(as_num(1).div(as_num(1))).toBeCloseTo(1);
  ex(as_num(1).div(as_num(-1))).toBeCloseTo(-1);
  ex(as_num(0.3).div(as_num(0.1))).toBeCloseTo(3);
});

test("num square root", () => {
  ex(as_num(1).sqrt()).toBeCloseTo(1);
  ex(as_num(4).sqrt()).toBeCloseTo(2);
  ex(as_num(0.25).sqrt()).toBeCloseTo(0.5);
});

test("num negation", () => {
  ex(as_num(1).neg()).toBeCloseTo(-1);
  ex(as_num(-1.2).neg()).toBeCloseTo(1.2);
  ex(as_num(0).neg()).toBeCloseTo(0);
});

test("num sign", () => {
  ex(as_num(10).sign()).toBeCloseTo(1);
  ex(as_num(-1.2).sign()).toBeCloseTo(-1);
  ex(as_num(0).sign()).toBeCloseTo(0);
});

test("num mod", () => {
  ex(as_num(10).mod(as_num(3))).toBeCloseTo(1);
  ex(as_num(10).mod(as_num(5))).toBeCloseTo(0);
  ex(as_num(10).mod(as_num(7))).toBeCloseTo(3);
});

test("num cos", () => {
  ex(as_num(0).cos()).toBeCloseTo(1);
  ex(as_num(Math.PI).cos()).toBeCloseTo(-1);
  ex(as_num(Math.PI / 2).cos()).toBeCloseTo(0);
});

test("num acos", () => {
  ex(as_num(1).acos()).toBeCloseTo(0);
  ex(as_num(-1).acos()).toBeCloseTo(Math.PI);
  ex(as_num(0).acos()).toBeCloseTo(Math.PI / 2);
});

test("num sin", () => {
  ex(as_num(0).sin()).toBeCloseTo(0);
  ex(as_num(Math.PI).sin()).toBeCloseTo(0);
  ex(as_num(Math.PI / 2).sin()).toBeCloseTo(1);
});

test("num asin", () => {
  ex(as_num(0).asin()).toBeCloseTo(0);
  ex(as_num(1).asin()).toBeCloseTo(Math.PI / 2);
  ex(as_num(-1).asin()).toBeCloseTo(-Math.PI / 2);
});

test("num tan", () => {
  ex(as_num(0).tan()).toBeCloseTo(0);
  ex(as_num(Math.PI).tan()).toBeCloseTo(0);
  ex(as_num(Math.PI / 4).tan()).toBeCloseTo(1);
});

test("num atan", () => {
  ex(as_num(0).atan()).toBeCloseTo(0);
  ex(as_num(1).atan()).toBeCloseTo(Math.PI / 4);
  ex(as_num(-1).atan()).toBeCloseTo(-Math.PI / 4);
});

test("num log", () => {
  ex(as_num(1).log()).toBeCloseTo(0);
  ex(as_num(Math.E).log()).toBeCloseTo(1);
  ex(as_num(10).log()).toBeCloseTo(Math.log(10));
});

test("num exp", () => {
  ex(as_num(0).exp()).toBeCloseTo(1);
  ex(as_num(1).exp()).toBeCloseTo(Math.E);
  ex(as_num(2).exp()).toBeCloseTo(Math.E ** 2);
});

test("num square", () => {
  ex(as_num(0).square()).toBeCloseTo(0);
  ex(as_num(1.5).square()).toBeCloseTo(2.25);
  ex(as_num(2).square()).toBeCloseTo(4);
});

test("num abs", () => {
  ex(as_num(0).abs()).toBeCloseTo(0);
  ex(as_num(1.5).abs()).toBeCloseTo(1.5);
  ex(as_num(-2).abs()).toBeCloseTo(2);
});

test("num compare", () => {
  ex(as_num(0).compare(0)).toBeCloseTo(0);
  ex(as_num(1).compare(0)).toBeCloseTo(1);
  ex(as_num(0).compare(1)).toBeCloseTo(-1);
});

test("num and", () => {
  ex(as_num(0).and(0)).toBeCloseTo(0);
  ex(as_num(1).and(0)).toBeCloseTo(0);
  ex(as_num(0).and(1)).toBeCloseTo(0);
  ex(as_num(1).and(1)).toBeCloseTo(1);
  ex(as_num(1).and(-1)).toBeCloseTo(-1);
});

test("num or", () => {
  ex(as_num(0).or(0)).toBeCloseTo(0);
  ex(as_num(1).or(0)).toBeCloseTo(1);
  ex(as_num(0).or(1)).toBeCloseTo(1);
  ex(as_num(1).or(1)).toBeCloseTo(1);
  ex(as_num(1).or(-1)).toBeCloseTo(1);
  ex(as_num(-1).or(1)).toBeCloseTo(-1);
});

test("num max", () => {
  ex(as_num(0).max(0)).toBeCloseTo(0);
  ex(as_num(1).max(0)).toBeCloseTo(1);
  ex(as_num(0).max(1)).toBeCloseTo(1);
  ex(as_num(1).max(1)).toBeCloseTo(1);
  ex(as_num(1).max(-1)).toBeCloseTo(1);
  ex(as_num(-1).max(1)).toBeCloseTo(1);
});

test("num min", () => {
  ex(as_num(0).min(0)).toBeCloseTo(0);
  ex(as_num(1).min(0)).toBeCloseTo(0);
  ex(as_num(0).min(1)).toBeCloseTo(0);
  ex(as_num(1).min(1)).toBeCloseTo(1);
  ex(as_num(1).min(-1)).toBeCloseTo(-1);
  ex(as_num(-1).min(1)).toBeCloseTo(-1);
});

test("num less than", () => {
  ex(as_num(0).lessThan(0)).toBeCloseTo(0);
  ex(as_num(1).lessThan(0)).toBeCloseTo(0);
  ex(as_num(0).lessThan(1)).toBeCloseTo(1);
  ex(as_num(1).lessThan(1)).toBeCloseTo(0);
  ex(as_num(1).lessThan(-1)).toBeCloseTo(0);
  ex(as_num(-1).lessThan(1)).toBeCloseTo(1);
});

test("num less than or equal", () => {
  ex(as_num(0).lessThanOrEqual(0)).toBeCloseTo(1);
  ex(as_num(1).lessThanOrEqual(0)).toBeCloseTo(0);
  ex(as_num(0).lessThanOrEqual(1)).toBeCloseTo(1);
  ex(as_num(1).lessThanOrEqual(1)).toBeCloseTo(1);
  ex(as_num(1).lessThanOrEqual(-1)).toBeCloseTo(0);
  ex(as_num(-1).lessThanOrEqual(1)).toBeCloseTo(1);
});

test("num greater than", () => {
  ex(as_num(0).greaterThan(0)).toBeCloseTo(0);
  ex(as_num(1).greaterThan(0)).toBeCloseTo(1);
  ex(as_num(0).greaterThan(1)).toBeCloseTo(0);
  ex(as_num(1).greaterThan(1)).toBeCloseTo(0);
  ex(as_num(1).greaterThan(-1)).toBeCloseTo(1);
  ex(as_num(-1).greaterThan(1)).toBeCloseTo(0);
});

test("num greater than or equal", () => {
  ex(as_num(0).greaterThanOrEqual(0)).toBeCloseTo(1);
  ex(as_num(1).greaterThanOrEqual(0)).toBeCloseTo(1);
  ex(as_num(0).greaterThanOrEqual(1)).toBeCloseTo(0);
  ex(as_num(1).greaterThanOrEqual(1)).toBeCloseTo(1);
  ex(as_num(1).greaterThanOrEqual(-1)).toBeCloseTo(1);
  ex(as_num(-1).greaterThanOrEqual(1)).toBeCloseTo(0);
});
test("num equals", () => {
  ex(as_num(0).equals(0)).toBeCloseTo(1);
  ex(as_num(1).equals(0)).toBeCloseTo(0);
  ex(as_num(0).equals(1)).toBeCloseTo(0);
  ex(as_num(1).equals(1)).toBeCloseTo(1);
  ex(as_num(1).equals(-1)).toBeCloseTo(0);
  ex(as_num(-1).equals(1)).toBeCloseTo(0);
});
