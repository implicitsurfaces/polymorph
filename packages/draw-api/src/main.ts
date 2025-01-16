import {
  ArcExtrusion2DNode,
  Box,
  Circle,
  ConeNode,
  ConeSurfaceNode,
  EasedWidthModulation,
  EllipseNode,
  LinearExtrusion2DNode,
  LinearWidthModulation,
  SphereNode,
  StaticWidthModulation,
} from "sketch";
import { point, angle, vector, distance } from "./geom";
export { point3D, vector3D, plane } from "./geom-3d";
import { ProfileEditor } from "./ProfileEditor";
import {
  AngleLike,
  asAngle,
  asDistance,
  DistanceLike,
  PointLike,
} from "./convert";
import { SolidEditor } from "./SolidEditor";

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

export function drawEllipse(
  majorRadius: DistanceLike,
  minorRadius: DistanceLike,
  center: PointLike | null = null,
): ProfileEditor {
  const ellipse = new ProfileEditor(
    new EllipseNode(asDistance(majorRadius), asDistance(minorRadius)),
  );
  if (!center) {
    return ellipse;
  }
  return ellipse.translate(center);
}

function parseModulation(
  startWidth: DistanceLike,
  endWidth?: DistanceLike,
  easing?: "in" | "out" | "inOut",
) {
  if (endWidth === undefined) {
    return new StaticWidthModulation(asDistance(startWidth));
  }

  const startDistance = asDistance(startWidth);
  const endDistance = asDistance(endWidth);

  if (!easing) {
    return new LinearWidthModulation(startDistance, endDistance);
  }

  return new EasedWidthModulation(startDistance, endDistance, easing);
}

export function drawLinearExtrusion(
  height: DistanceLike,
  startWidth: DistanceLike,
  endWidth?: DistanceLike,
  easing?: "in" | "out" | "inOut",
): ProfileEditor {
  const modulation = parseModulation(startWidth, endWidth, easing);

  return new ProfileEditor(
    new LinearExtrusion2DNode(asDistance(height), modulation),
  );
}

export function drawArcExtrusion(
  radius: DistanceLike,
  angle: AngleLike,
  startWidth: DistanceLike,
  endWidth?: DistanceLike,
  easing?: "in" | "out" | "inOut",
): ProfileEditor {
  const modulation = parseModulation(startWidth, endWidth, easing);
  return new ProfileEditor(
    new ArcExtrusion2DNode(asDistance(radius), asAngle(angle), modulation),
  );
}

export function makeSphere(radius: DistanceLike): SolidEditor {
  return new SolidEditor(new SphereNode(asDistance(radius)));
}

export function makeCone(
  radius: DistanceLike,
  height: DistanceLike,
  surfaceOnly = false,
): SolidEditor {
  const node = surfaceOnly
    ? new ConeSurfaceNode(asDistance(radius), asDistance(height))
    : new ConeNode(asDistance(radius), asDistance(height));
  return new SolidEditor(node);
}
