import { SelectTool } from "./SelectTool.ts";
import { PointTool } from "./PointTool.ts";

export function allTools() {
  return [new SelectTool(), new PointTool()];
}
