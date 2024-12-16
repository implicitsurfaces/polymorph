import {
  Box,
  Circle,
  LinearExtrusion2DNode,
  LinearWidthModulation,
  StaticWidthModulation,
} from "sketch";
import { point, angle, vector, distance } from "./geom";
import { ProfileEditor } from "./ProfileEditor";
import { asDistance, DistanceLike, PointLike } from "./convert";

export type { Point, Vector, Distance, Angle } from "./geom";
export type { ProfileEditor } from "./ProfileEditor";

export { draw } from "./draw";

export { point, angle, vector, distance };
export {
  realVar,
  pointVar,
  angleVar,
  distanceVar,
  pointVarPolar,
  pointVarCartesian,
} from "./variables";
export type { DistanceLike, PointLike, AngleLike, VectorLike } from "./convert";

export { LossFunction } from "./loss";

export function drawCircle(
  radius: DistanceLike,
  center: PointLike | null = null,
): ProfileEditor {
  const circle = new ProfileEditor(new Circle(asDistance(radius)));
  if (!center) {
    return circle;
  }
  return circle.translate(center);
}

export function drawBox(
  width: DistanceLike,
  height: DistanceLike,
  center: PointLike | null = null,
): ProfileEditor {
  const box = new ProfileEditor(new Box(asDistance(width), asDistance(height)));
  if (!center) {
    return box;
  }
  return box.translate(center);
}

export function drawLinearExtrusion(
  height: DistanceLike,
  startWidth: DistanceLike,
  endWidth?: DistanceLike,
): ProfileEditor {
  let modulation: LinearWidthModulation | StaticWidthModulation;

  if (endWidth === undefined) {
    modulation = new StaticWidthModulation(asDistance(startWidth));
  } else {
    modulation = new LinearWidthModulation(
      asDistance(startWidth),
      asDistance(endWidth),
    );
  }

  return new ProfileEditor(
    new LinearExtrusion2DNode(asDistance(height), modulation),
  );
}
