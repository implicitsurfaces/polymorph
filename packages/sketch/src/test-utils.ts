import { expect } from "vitest";
import { simple_eval } from "./num-tree";
import { as_vec } from "./geom";
import { Num } from "./num";
import { DistField } from "./types";

export function ex(num: Num): ReturnType<typeof expect> {
  return expect(simple_eval(num.n));
}

const FILLED_CHAR = "█";
const EMPTY_CHAR = " ";

type ImageData = boolean[][];
function toASCII(imageData: ImageData): string {
  return imageData
    .map((row) =>
      row.map((pixel) => (pixel ? FILLED_CHAR : EMPTY_CHAR)).join(""),
    )
    .join("\n");
}

const GRID_SIZE = 50;
const scalingFactor = GRID_SIZE / 2;
const GRID = Array.from({ length: GRID_SIZE }, (_, i) =>
  Array.from({ length: GRID_SIZE * 2 }, (_, j) =>
    as_vec(
      (j / 2 - GRID_SIZE / 2) / scalingFactor,
      (i - GRID_SIZE / 2) / scalingFactor,
    ).point_from_origin(),
  ),
);

export function expect_ascii_dist(distField: DistField) {
  const imageData = GRID.map((row) =>
    row.map((point) => simple_eval(distField.distance_to(point).n) < 0),
  );
  return expect(toASCII(imageData));
}
