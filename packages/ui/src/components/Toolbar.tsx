import { useContext, useMemo } from "react";

import { CurrentToolContext } from "./CurrentTool";
import { DocumentManagerContext } from "../doc/DocumentManagerContext";

import { Action, TriggerAction } from "../actions/Action";
import { Tool } from "../tools/Tool";

import { Menu, ActionItem } from "./DropdownMenu";
import menuIcon from "../assets/tool-icons/menu.svg";

import "./Toolbar.css";

interface ToolbarMenuProps {
  actions: Action[];
}

export function ToolbarMenu({ actions }: ToolbarMenuProps) {
  // Get which actions should appear in the menu. For now, we consider them
  // to be those without an icon.
  //
  const actionsWithMenuItem = useMemo<Action[]>(() => {
    const res = actions.filter((action) => action.menu !== undefined);
    res.sort((a, b) => a.menuIndex - b.menuIndex);
    return res;
  }, [actions]);

  return (
    <Menu trigger={<img src={menuIcon} />}>
      {actionsWithMenuItem.map((action) => (
        <ActionItem action={action} key={action.name} />
      ))}
    </Menu>
  );
}

interface ToolbarProps {
  actions: Action[];
}

export function Toolbar({ actions }: ToolbarProps) {
  const { documentManager } = useContext(DocumentManagerContext);
  const { currentTool, setCurrentTool } = useContext(CurrentToolContext);

  // Get which actions should appear in the toolbar. For now, we consider them
  // to be those with an icon. In the future, this function could also sort
  // them in a different order (e.g., `action.toolbarIndex`).
  //
  const actionsWithIcon = useMemo<Action[]>(() => {
    return actions.filter((action) => action.icon !== undefined);
  }, [actions]);

  function onClick(action: Action) {
    if (action instanceof Tool) {
      setCurrentTool(action);
    } else if (action instanceof TriggerAction) {
      action.onTrigger(documentManager);
    }
  }

  function getItem(action: Action) {
    const name = action.name;
    const hint = action.shortcut
      ? `${name} (${action.shortcut.prettyStr})`
      : name;
    const icon = action.icon ?? "";
    return (
      <img
        className={action === currentTool ? "is-active" : ""}
        src={icon}
        alt={hint}
        title={hint}
        key={name}
        onClick={() => onClick(action)}
      />
    );
  }

  return (
    <div className="toolbar">
      <ToolbarMenu actions={actions} />
      {actionsWithIcon.map((action) => getItem(action))}
    </div>
  );
}

export default Toolbar;
