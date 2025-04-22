import * as Actions from "../actions";
import * as Tools from "../tools";

export const actions = {
  Undo: new Actions.UndoAction(),
  Redo: new Actions.RedoAction(),

  Open: new Actions.OpenAction(),
  Save: new Actions.SaveAction(),
  SaveAs: new Actions.SaveAsAction(),

  AddDistance: new Actions.AddDistanceAction(),

  SelectTool: new Tools.SelectTool(),
  PointTool: new Tools.PointTool(),
  LineSegmentTool: new Tools.LineSegmentTool(),
};
