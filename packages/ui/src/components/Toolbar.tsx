import { ReactNode, useContext } from "react";

import { CurrentToolContext } from "./CurrentTool";
import { DocumentManagerContext } from "../doc/DocumentManagerContext";

import { Action, TriggerAction } from "../actions/Action";
import { Tool } from "../tools/Tool";

import { Menu } from "./DropdownMenu";
import menuIcon from "../assets/tool-icons/menu.svg";

import "./Toolbar.css";

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
  const { documentManager } = useContext(DocumentManagerContext);
  const { currentTool, setCurrentTool } = useContext(CurrentToolContext);

  const name = action.name;
  const hint = action.shortcut
    ? `${name} (${action.shortcut.prettyStr})`
    : name;
  const icon = action.icon ?? "";

  function onClick(action: Action) {
    if (action instanceof Tool) {
      setCurrentTool(action);
    } else if (action instanceof TriggerAction) {
      action.onTrigger(documentManager);
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
