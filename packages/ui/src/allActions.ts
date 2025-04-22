import { Action } from "./actions/Action";

import { UndoAction, RedoAction } from "./actions/UndoRedoActions";
import {
  OpenAction,
  SaveAction,
  SaveAsAction,
} from "./actions/OpenSaveActions";
import { AddDistanceAction } from "./actions/AddDistanceAction";

import { SelectTool } from "./tools/SelectTool";
import { PointTool } from "./tools/PointTool";
import { LineSegmentTool } from "./tools/LineSegmentTool";

export function allActions(): Action[] {
  return [
    new UndoAction(),
    new RedoAction(),
    new OpenAction(),
    new SaveAction(),
    new SaveAsAction(),
    new SelectTool(),
    new PointTool(),
    new LineSegmentTool(),
    new AddDistanceAction(),
  ];
}
