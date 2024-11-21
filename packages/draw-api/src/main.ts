import { Circle, DistanceNode, PointNode } from "sketch";
import { point, angle, vector } from "./geom";
import { ProfileEditor } from "./ProfileEditor";
import { asDistance } from "./convert";

export { draw } from "./draw";

export { point, angle, vector };

export function drawCircle(
  radius: number | DistanceNode,
  center: [number, number] | PointNode | null = null,
): ProfileEditor {
  const circle = new ProfileEditor(new Circle(asDistance(radius)));
  if (!center) {
    return circle;
  }
  return circle.translate(center);
}
