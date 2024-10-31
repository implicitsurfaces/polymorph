import { test } from "vitest";

import { ex } from "./test-utils";

import {
  add,
  sub,
  mul,
  div,
  mod,
  max,
  min,
  atan2,
  compare,
  and,
  or,
  less_than,
  less_than_or_equal,
  greater_than,
  greater_than_or_equal,
  if_non_zero_else,
  hypot,
  clamp,
} from "./num-ops";

test("add", () => {
  ex(add(1, 3)).toBeCloseTo(4);
  ex(add(1, 0)).toBeCloseTo(1);
  ex(add(1, -1)).toBeCloseTo(0);
  ex(add(0.3, 0.1)).toBeCloseTo(0.4);
});

test("sub", () => {
  ex(sub(1, 3)).toBeCloseTo(-2);
  ex(sub(1, 0)).toBeCloseTo(1);
  ex(sub(1, -1)).toBeCloseTo(2);
  ex(sub(0.3, 0.1)).toBeCloseTo(0.2);
});

test("mul", () => {
  ex(mul(1, 3)).toBeCloseTo(3);
  ex(mul(1, 0)).toBeCloseTo(0);
  ex(mul(1, -1)).toBeCloseTo(-1);
  ex(mul(0.3, 0.1)).toBeCloseTo(0.03);
});

test("div", () => {
  ex(div(1, 3)).toBeCloseTo(1 / 3);
  ex(div(1, 1)).toBeCloseTo(1);
  ex(div(1, -1)).toBeCloseTo(-1);
  ex(div(0.3, 0.1)).toBeCloseTo(3);
});

test("mod", () => {
  ex(mod(10, 3)).toBeCloseTo(1);
  ex(mod(10, 2)).toBeCloseTo(0);
  ex(mod(10, 5)).toBeCloseTo(0);
  ex(mod(10, 4)).toBeCloseTo(2);
});

test("max", () => {
  ex(max(1, 3)).toBeCloseTo(3);
  ex(max(1, 0)).toBeCloseTo(1);
  ex(max(1, -1)).toBeCloseTo(1);
  ex(max(0.3, 0.1)).toBeCloseTo(0.3);
  ex(max(0.3, 0.1, -2, 12, -123.1)).toBeCloseTo(12);
});

test("min", () => {
  ex(min(1, 3)).toBeCloseTo(1);
  ex(min(1, 0)).toBeCloseTo(0);
  ex(min(1, -1)).toBeCloseTo(-1);
  ex(min(0.3, 0.1)).toBeCloseTo(0.1);
  ex(min(0.3, 0.1, -2, -123.1)).toBeCloseTo(-123.1);
});

test("atan2", () => {
  ex(atan2(1, 1)).toBeCloseTo(Math.PI / 4);
  ex(atan2(1, 0)).toBeCloseTo(Math.PI / 2);
  ex(atan2(1, -1)).toBeCloseTo((3 * Math.PI) / 4);
  ex(atan2(0, 1)).toBeCloseTo(0);
  ex(atan2(0, -1)).toBeCloseTo(Math.PI);
  ex(atan2(-1, 1)).toBeCloseTo(-Math.PI / 4);
  ex(atan2(-1, 0)).toBeCloseTo(-Math.PI / 2);
  ex(atan2(-1, -1)).toBeCloseTo((-3 * Math.PI) / 4);
});

test("compare", () => {
  ex(compare(1, 3)).toBeCloseTo(-1);
  ex(compare(1, 1)).toBeCloseTo(0);
  ex(compare(1, -1)).toBeCloseTo(1);
  ex(compare(0.3, 0.1)).toBeCloseTo(1);
  ex(compare(0.1, 0.3)).toBeCloseTo(-1);
});

test("and", () => {
  ex(and(1, 3)).toBeCloseTo(3);
  ex(and(1, 0)).toBeCloseTo(0);
  ex(and(0, 1)).toBeCloseTo(0);
  ex(and(0, 0)).toBeCloseTo(0);
  ex(and(0.3, 0.1)).toBeCloseTo(0.1);
  ex(and(-2, 1)).toBeCloseTo(1);
  ex(and(-2, 1, 0, 12)).toBeCloseTo(0);
  ex(and(-2, 1, 5, 12)).toBeCloseTo(12);
});

test("or", () => {
  ex(or(1, 3)).toBeCloseTo(1);
  ex(or(1, 0)).toBeCloseTo(1);
  ex(or(0, 1)).toBeCloseTo(1);
  ex(or(0, 0)).toBeCloseTo(0);
  ex(or(0.3, 0.1)).toBeCloseTo(0.3);
  ex(or(-2, 1)).toBeCloseTo(-2);
  ex(or(-2, 0)).toBeCloseTo(-2);
  ex(or(0, 0, 0, 0, 1)).toBeCloseTo(1);
  ex(or(0, 0, 2, 0, 1)).toBeCloseTo(2);
});

test("less_than", () => {
  ex(less_than(1, 3)).toBeCloseTo(1);
  ex(less_than(1, 1)).toBeCloseTo(0);
  ex(less_than(1, -1)).toBeCloseTo(0);
  ex(less_than(0.3, 0.1)).toBeCloseTo(0);
  ex(less_than(0.1, 0.3)).toBeCloseTo(1);
  ex(less_than(-2, -1)).toBeCloseTo(1);
  ex(less_than(-1, -2)).toBeCloseTo(0);
  ex(less_than(-2, -2)).toBeCloseTo(0);
});

test("less_than_or_equal", () => {
  ex(less_than_or_equal(1, 3)).toBeCloseTo(1);
  ex(less_than_or_equal(1, 1)).toBeCloseTo(1);
  ex(less_than_or_equal(1, -1)).toBeCloseTo(0);
  ex(less_than_or_equal(0.3, 0.1)).toBeCloseTo(0);
  ex(less_than_or_equal(0.1, 0.3)).toBeCloseTo(1);
  ex(less_than_or_equal(-2, -1)).toBeCloseTo(1);
  ex(less_than_or_equal(-1, -2)).toBeCloseTo(0);
  ex(less_than_or_equal(-2, -2)).toBeCloseTo(1);
});

test("greater_than", () => {
  ex(greater_than(1, 3)).toBeCloseTo(0);
  ex(greater_than(1, 1)).toBeCloseTo(0);
  ex(greater_than(1, -1)).toBeCloseTo(1);
  ex(greater_than(0.3, 0.1)).toBeCloseTo(1);
  ex(greater_than(0.1, 0.3)).toBeCloseTo(0);
  ex(greater_than(-2, -1)).toBeCloseTo(0);
  ex(greater_than(-1, -2)).toBeCloseTo(1);
  ex(greater_than(-2, -2)).toBeCloseTo(0);
});

test("greater_than_or_equal", () => {
  ex(greater_than_or_equal(1, 3)).toBeCloseTo(0);
  ex(greater_than_or_equal(1, 1)).toBeCloseTo(1);
  ex(greater_than_or_equal(1, -1)).toBeCloseTo(1);
  ex(greater_than_or_equal(0.3, 0.1)).toBeCloseTo(1);
  ex(greater_than_or_equal(0.1, 0.3)).toBeCloseTo(0);
  ex(greater_than_or_equal(-2, -1)).toBeCloseTo(0);
  ex(greater_than_or_equal(-1, -2)).toBeCloseTo(1);
  ex(greater_than_or_equal(-2, -2)).toBeCloseTo(1);
});

test("if_non_zero_else", () => {
  ex(if_non_zero_else(1, 3, 4)).toBeCloseTo(3);
  ex(if_non_zero_else(0, 3, 4)).toBeCloseTo(4);
  ex(if_non_zero_else(-1, 3, 4)).toBeCloseTo(3);
  ex(if_non_zero_else(0.3, 0.1, 0.2)).toBeCloseTo(0.1);
  ex(if_non_zero_else(0.1, 0.1, 0.2)).toBeCloseTo(0.1);
  ex(if_non_zero_else(0.1, 0.2, 0.1)).toBeCloseTo(0.2);
  ex(if_non_zero_else(-0.1, 0.2, 0.1)).toBeCloseTo(0.2);
  ex(if_non_zero_else(-0, 0.1, 0.2)).toBeCloseTo(0.2);
  ex(if_non_zero_else(0, 10000, 0.2)).toBeCloseTo(0.2);
});

test("hypot", () => {
  ex(hypot(3, 4)).toBeCloseTo(5);
  ex(hypot(1, 1)).toBeCloseTo(Math.sqrt(2));
  ex(hypot(1, 0)).toBeCloseTo(1);
  ex(hypot(0, 1)).toBeCloseTo(1);
  ex(hypot(0, 0)).toBeCloseTo(0);
});

test("clamp", () => {
  ex(clamp(1, 3, 4)).toBeCloseTo(3);
  ex(clamp(5, 3, 4)).toBeCloseTo(4);
  ex(clamp(3.5, 3, 4)).toBeCloseTo(3.5);
  ex(clamp(3, 3, 4)).toBeCloseTo(3);
  ex(clamp(4, 3, 4)).toBeCloseTo(4);
  ex(clamp(1, 3, 3)).toBeCloseTo(3);
  ex(clamp(5, 3, 3)).toBeCloseTo(3);
  ex(clamp(3.5, 3, 3)).toBeCloseTo(3);
  ex(clamp(3, 3, 3)).toBeCloseTo(3);
  ex(clamp(4, 3, 3)).toBeCloseTo(3);
});
