import { SelectTool } from "./SelectTool.ts";
import { PointTool } from "./PointTool.ts";

export function makeTools() {
  return [new SelectTool(), new PointTool()];
}
