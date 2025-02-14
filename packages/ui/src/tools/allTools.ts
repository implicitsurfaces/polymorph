import { Tool } from "./Tool.ts";

import { SelectTool } from "./SelectTool.ts";
import { PointTool } from "./PointTool.ts";
import { LineSegmentTool } from "./LineSegmentTool.ts";

export function allTools(): Tool[] {
  return [new SelectTool(), new PointTool(), new LineSegmentTool()];
}
