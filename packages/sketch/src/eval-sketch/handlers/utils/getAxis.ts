import { UnitVec3, X_AXIS, Y_AXIS, Z_AXIS } from "../../../geom-3d";
import { Vector3Node } from "../../../sketch-nodes";
import { guardVec3 } from "../../guards";

export function getAxis(
  axis: "x" | "y" | "z" | Vector3Node,
  inputAxis: unknown,
): UnitVec3 {
  if (axis === "x") {
    return X_AXIS;
  }
  if (axis === "y") {
    return Y_AXIS;
  }
  if (axis === "z") {
    return Z_AXIS;
  }
  if (axis instanceof Vector3Node) {
    return guardVec3(inputAxis).normalize();
  }

  throw new Error(`Unknown axis: ${axis}`);
}
