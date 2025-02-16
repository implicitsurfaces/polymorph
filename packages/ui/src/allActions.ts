import { Action } from "./actions/Action.ts";

import { UndoAction, RedoAction } from "./actions/UndoRedoActions.ts";
import { AddDistanceAction } from "./actions/AddDistanceAction.ts";

import { SelectTool } from "./tools/SelectTool.ts";
import { PointTool } from "./tools/PointTool.ts";
import { LineSegmentTool } from "./tools/LineSegmentTool.ts";

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
