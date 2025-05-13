import { ReactNode } from "react";

import { useCurrentToolContext } from "./CurrentTool";
import { useDocumentManager } from "../doc/DocumentManagerContext";

import { Action, TriggerAction } from "../actions/Action";
import { Tool } from "../tools/Tool";

import { Menu } from "./DropdownMenu";
import menuIcon from "../assets/tool-icons/menu.svg";

import "./Toolbar.css";
import { useViewContext } from "../view";

interface ToolbarMenuProps {
  children: ReactNode;
}

export function ToolbarMenu({ children }: ToolbarMenuProps) {
  return <Menu trigger={<img src={menuIcon} />}>{children}</Menu>;
}

interface ToolbarActionItemProps {
  action: Action;
}

export function ToolbarActionItem({ action }: ToolbarActionItemProps) {
  const documentManager = useDocumentManager();
  const { currentTool, setCurrentTool } = useCurrentToolContext();
  const viewContext = useViewContext();

  const name = action.name;
  const hint = action.shortcut
    ? `${name} (${action.shortcut.prettyStr})`
    : name;
  const icon = action.icon ?? "";

  function onClick(action: Action) {
    if (action instanceof Tool) {
      setCurrentTool(action);
    } else if (action instanceof TriggerAction) {
      action.onTrigger(documentManager, viewContext);
    }
  }

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

interface ToolbarProps {
  children: ReactNode;
}

export function Toolbar({ children }: ToolbarProps) {
  return <div className="toolbar">{children}</div>;
}

export default Toolbar;
