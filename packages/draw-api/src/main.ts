import { Box, Circle } from "sketch";
import { point, angle, vector, distance } from "./geom";
import { ProfileEditor } from "./ProfileEditor";
import { asDistance, DistanceLike, PointLike } from "./convert";

export { draw } from "./draw";

export { point, angle, vector, distance };

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
