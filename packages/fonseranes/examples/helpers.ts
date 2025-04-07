import { XY_Plane, renderAsSVG, ShapeInput } from "../src/main";
import fs from "node:fs";

export const outputSVG = (
  shape: ShapeInput | ShapeInput[],
  filename = "test.svg",
  threshold = 0,
  precision = 1,
) => {
  const svg = renderAsSVG(shape, { threshold, cellFactor: precision });
  fs.writeFileSync(filename, svg);
};

export function sliceToHeight(
  shape: any,
  height: number,
  steps: number,
  basePlane = XY_Plane,
) {
  const step = height / steps;
  const slices: ShapeInput[] = [shape.slice(basePlane)];
  for (let i = step; i < height; i += step) {
    slices.push({
      shape: shape.slice(basePlane.translateZ(i)),
      strokeWidth: 0.4,
    });
  }

  slices.push(shape.slice(basePlane.translateZ(height)));

  return slices;
}
