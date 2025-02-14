import { useContext } from "react";

import { CurrentToolContext } from "./CurrentTool.ts";
import { Tool } from "./Tool.ts";

import { Action } from "../actions/Action.ts";

import { DocumentManager } from "../DocumentManager.ts";

import "./Toolbar.css";

interface ToolbarProps {
  tools: Array<Tool>;
  actions: Action[];
  documentManager: DocumentManager;
}

export function Toolbar({ tools, actions, documentManager }: ToolbarProps) {
  const { currentTool, setCurrentTool } = useContext(CurrentToolContext);

  return (
    <div className="toolbar">
      {tools.map((tool) => {
        const name = tool.name;
        return (
          <img
            className={tool === currentTool ? "is-active" : ""}
            src={tool.icon}
            alt={name}
            title={name}
            key={name}
            onClick={() => {
              setCurrentTool(tool);
            }}
          />
        );
      })}
      {actions.map((action) => {
        const name = action.name;
        const hint = `${name} (${action.shortcut.prettyStr})`;
        return (
          <img
            src={action.icon}
            alt={hint}
            title={hint}
            key={name}
            onClick={() => {
              if (action.onTrigger) {
                action.onTrigger(documentManager);
              }
            }}
          />
        );
      })}
    </div>
  );
}

export default Toolbar;
