import { useContext, useMemo } from "react";

import { CurrentToolContext } from "./CurrentTool";
import { Tool } from "./Tool";

import { Action, TriggerAction } from "../actions/Action";

import { DocumentManager } from "../doc/DocumentManager";

import "./Toolbar.css";

interface ToolbarProps {
  actions: Action[];
  documentManager: DocumentManager;
}

export function Toolbar({ actions, documentManager }: ToolbarProps) {
  const { currentTool, setCurrentTool } = useContext(CurrentToolContext);

  // Get which actions should appear in the toolbar. For now, we consider them
  // to be those with an icon. In the future, this function could also sort
  // them in a different order (e.g., `action.toolbarIndex`).
  //
  const actions_ = useMemo<Action[]>(() => {
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
    <div className="toolbar">{actions_.map((action) => getItem(action))}</div>
  );
}

export default Toolbar;
