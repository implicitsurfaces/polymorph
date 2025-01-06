import { expect } from "vitest";
import { simpleEval, dedupeEval, naiveEval } from "./num-tree";
import { Point, asVec } from "./geom";
import { Num, ONE, ZERO } from "./num";
import { DistField, Segment, SolidDistField } from "./types";
import { fidgetRender } from "./num-tree-fidget";
import { embedPoint, Point3D, XZ_PLANE } from "./geom-3d";

export function ex(num: Num) {
  return expect(simpleEval(num.n));
}

export function exVar(num: Num, vars: Record<string, number>) {
  return expect(naiveEval(num.n, new Map(Object.entries(vars))));
}

const FILLED_CHAR = "█";
const EMPTY_CHAR = " ";

type BooleanImageData = boolean[][];
function booleansToASCII(imageData: BooleanImageData, double = false): string {
  const fillChar = double ? FILLED_CHAR + FILLED_CHAR : FILLED_CHAR;
  const emptyChar = double ? EMPTY_CHAR + EMPTY_CHAR : EMPTY_CHAR;

  return imageData
    .map((row) => row.map((pixel) => (pixel ? fillChar : emptyChar)).join(""))
    .join("\n");
}

function intArrayToImageData(imageData: Uint8Array): BooleanImageData {
  const rowLength = Math.sqrt(imageData.length);
  const result: BooleanImageData = [];
  for (let i = 0; i < imageData.length; i += rowLength) {
    const row = [...imageData.slice(i, i + rowLength)].map(
      (value) => value > 0,
    );
    result.push(row);
  }
  return result;
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

async function callOnGrid<T>(fn: (point: Point) => Promise<T>) {
  const results = [];
  for (const row of GRID) {
    const rowResults = await Promise.all(row.map(fn));
    results.push(rowResults);
  }
  return results;
}

export async function expectASCIIshape(distField: DistField) {
  const imageData = await callOnGrid(async (point): Promise<boolean> => {
    const dist = await dedupeEval(distField.distanceTo(point).n);
    return dist < 0;
  });
  return expect(booleansToASCII(imageData));
}

export async function expectASCIISolidAngle(segment: Segment) {
  const imageData = await callOnGrid((point) =>
    dedupeEval(segment.solidAngle(point).turns.n),
  );

  return expect(gradientToASCII(imageData));
}

export async function expectASCIIDistance(distField: DistField) {
  const imageData = await callOnGrid((point) =>
    dedupeEval(distField.distanceTo(point).n),
  );
  return expect(gradientToASCII(imageData, 0.5));
}

export async function expectASCIIHeightMap(distField: SolidDistField) {
  const plane = XZ_PLANE.translateTo(new Point3D(ZERO, ONE.neg(), ZERO));
  const imageData = await callOnGrid((point) =>
    dedupeEval(distField.valueAt(embedPoint(point, plane)).n),
  );
  return expect(gradientToASCII(imageData, 0.5));
}

export async function expectFidgetRender(distField: DistField) {
  const render = await fidgetRender(distField);
  const imageData = intArrayToImageData(render);
  return expect(booleansToASCII(imageData, true));
}
