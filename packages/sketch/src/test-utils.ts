import { expect } from "vitest";
import { simple_eval } from "./num-tree";
import { asVec } from "./geom";
import { Num } from "./num";
import { DistField, Segment } from "./types";

export function ex(num: Num) {
  return expect(simple_eval(num.n));
}

const FILLED_CHAR = "█";
const EMPTY_CHAR = " ";

type BooleanImageData = boolean[][];
function booleansToASCII(imageData: BooleanImageData): string {
  return imageData
    .map((row) =>
      row.map((pixel) => (pixel ? FILLED_CHAR : EMPTY_CHAR)).join(""),
    )
    .join("\n");
}

const POSITIVE_GRADIENT_SCALE = ["˖", "░", "▒", "▓", "█"];
const NEGATIVE_GRADIENT_SCALE = ["-", "○", "◎", "◉", "●"];
type GradientImageData = number[][];
function gradientToASCII(imageData: GradientImageData, max = 1): string {
  return imageData
    .map((row) =>
      row
        .map((value) => {
          if (Math.abs(value) < 0) {
            return " ";
          }
          const scale =
            value > 0 ? POSITIVE_GRADIENT_SCALE : NEGATIVE_GRADIENT_SCALE;
          const index = Math.min(
            Math.floor((Math.abs(value) * scale.length) / max),
            scale.length - 1,
          );
          return scale[index];
        })
        .join(""),
    )
    .join("\n");
}

const GRID_SIZE = 50;
const scalingFactor = GRID_SIZE / 2;
const GRID = Array.from({ length: GRID_SIZE }, (_, i) =>
  Array.from({ length: GRID_SIZE * 2 }, (_, j) =>
    asVec(
      (j / 2 - GRID_SIZE / 2) / scalingFactor,
      (GRID_SIZE / 2 - i) / scalingFactor,
    ).pointFromOrigin(),
  ),
);

export function expectASCIIshape(distField: DistField) {
  const imageData = GRID.map((row) =>
    row.map((point) => simple_eval(distField.distanceTo(point).n) < 0),
  );
  return expect(booleansToASCII(imageData));
}

export function expectASCIISolidAngle(segment: Segment) {
  const imageData = GRID.map((row) =>
    row.map((point) => simple_eval(segment.solidAngle(point).turns.n)),
  );
  return expect(gradientToASCII(imageData));
}

export function expectASCIIDistance(distField: DistField) {
  const imageData = GRID.map((row) =>
    row.map((point) => simple_eval(distField.distanceTo(point).n)),
  );
  return expect(gradientToASCII(imageData, 0.5));
}
