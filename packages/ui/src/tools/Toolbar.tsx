import { useContext } from "react";

import { CurrentToolContext } from "./CurrentTool.ts";
import { Tool } from "./Tool.ts";

import "./Toolbar.css";

interface ToolbarProps {
  tools: Array<Tool>;
}

export function Toolbar({ tools }: ToolbarProps) {
  const { currentTool, setCurrentTool } = useContext(CurrentToolContext);

  return (
    <div className="toolbar">
      {tools.map((tool) => {
        const name = tool.name;
        return (
          <img
            className={name === currentTool ? "is-active" : ""}
            src={tool.icon}
            alt={name}
            title={name}
            key={name}
            onClick={() => {
              setCurrentTool(name);
            }}
          />
        );
      })}
    </div>
  );
}

export default Toolbar;
