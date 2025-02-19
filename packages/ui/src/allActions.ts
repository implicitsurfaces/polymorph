import { Action } from "./actions/Action";

import { UndoAction, RedoAction } from "./actions/UndoRedoActions";
import { AddDistanceAction } from "./actions/AddDistanceAction";

import { SelectTool } from "./tools/SelectTool";
import { PointTool } from "./tools/PointTool";
import { LineSegmentTool } from "./tools/LineSegmentTool";

export function allActions(): Action[] {
  return [
    new UndoAction(),
    new RedoAction(),
    new SelectTool(),
    new PointTool(),
    new LineSegmentTool(),
    new AddDistanceAction(),
  ];
}
