import { MenuActionItem } from "../components/DropdownMenu";
import { Toolbar, ToolbarMenu, ToolbarActionItem } from "../components/Toolbar";
import { actions } from "./AppActions";

export function AppToolbar() {
  return (
    <Toolbar>
      <ToolbarMenu>
        <MenuActionItem action={actions.Undo} />
        <MenuActionItem action={actions.Redo} />
        <MenuActionItem action={actions.Open} />
        <MenuActionItem action={actions.Save} />
        <MenuActionItem action={actions.SaveAs} />
        <MenuActionItem action={actions.ToggleSideBySideCanvas} />
      </ToolbarMenu>
      <ToolbarActionItem action={actions.SelectTool} />
      <ToolbarActionItem action={actions.PointTool} />
      <ToolbarActionItem action={actions.LineSegmentTool} />
      <ToolbarActionItem action={actions.AddDistance} />
    </Toolbar>
  );
}
