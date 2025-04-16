import { useContext, useMemo } from "react";

import { CurrentToolContext } from "./CurrentTool";

import { Action, TriggerAction } from "../actions/Action";
import { Tool } from "../tools/Tool";

import { DocumentManager } from "../doc/DocumentManager";

import * as DropdownMenu from "./DropdownMenu";
import menuIcon from "../assets/tool-icons/menu.svg";

import "./Toolbar.css";

interface ToolbarProps {
  actions: Action[];
  documentManager: DocumentManager;
}

export function ToolbarMenu({ actions, documentManager }: ToolbarProps) {
  // Get which actions should appear in the menu. For now, we consider them
  // to be those without an icon.
  //
  const actionsWithMenuItem = useMemo<Action[]>(() => {
    const res = actions.filter((action) => action.menu !== undefined);
    res.sort((a, b) => a.menuIndex - b.menuIndex);
    return res;
  }, [actions]);

  function onClick(action: Action) {
    if (action instanceof TriggerAction) {
      action.onTrigger(documentManager);
    }
  }

  function getShortcutSlot(action: Action) {
    if (!action.shortcut) {
      return undefined;
    }
    return <div className="right-slot">{action.shortcut.prettyStr}</div>;
  }

  function getItem(action: Action) {
    return (
      <DropdownMenu.Item key={action.name} onClick={() => onClick(action)}>
        {action.name} {getShortcutSlot(action)}
      </DropdownMenu.Item>
    );
  }

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger>
        <img src={menuIcon} />
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="dropdown-menu"
          alignOffset={-8}
          align="start"
          sideOffset={10}
        >
          {actionsWithMenuItem.map((action) => getItem(action))}
          <DropdownMenu.Arrow className="arrow" />
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

export function Toolbar({ actions, documentManager }: ToolbarProps) {
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
      <ToolbarMenu actions={actions} documentManager={documentManager} />
      {actionsWithIcon.map((action) => getItem(action))}
    </div>
  );
}

export default Toolbar;
