import { test } from "vitest";

import { ex } from "./test-utils";

import {
  max,
  min,
  atan2,
  ifTruthyElse,
  hypot,
  clamp,
  sigmoid,
} from "./num-ops";

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

test("if_non_zero_else", () => {
  ex(ifTruthyElse(1, 3, 4)).toBeCloseTo(3);
  ex(ifTruthyElse(0, 3, 4)).toBeCloseTo(4);
  ex(ifTruthyElse(-1, 3, 4)).toBeCloseTo(3);
  ex(ifTruthyElse(0.3, 0.1, 0.2)).toBeCloseTo(0.1);
  ex(ifTruthyElse(0.1, 0.1, 0.2)).toBeCloseTo(0.1);
  ex(ifTruthyElse(0.1, 0.2, 0.1)).toBeCloseTo(0.2);
  ex(ifTruthyElse(-0.1, 0.2, 0.1)).toBeCloseTo(0.2);
  ex(ifTruthyElse(-0, 0.1, 0.2)).toBeCloseTo(0.2);
  ex(ifTruthyElse(0, 10000, 0.2)).toBeCloseTo(0.2);
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

test("sigmoid", () => {
  ex(sigmoid(0)).toBeCloseTo(0.5);
  ex(sigmoid(1)).toBeCloseTo(1 / (1 + Math.exp(-1)));
  ex(sigmoid(-1)).toBeCloseTo(1 / (1 + Math.exp(1)));
  ex(sigmoid(3)).toBeCloseTo(1 / (1 + Math.exp(-3)));
  ex(sigmoid(-3)).toBeCloseTo(1 / (1 + Math.exp(3)));
  ex(sigmoid(10)).toBeCloseTo(1);
  ex(sigmoid(-10)).toBeCloseTo(0);
});
