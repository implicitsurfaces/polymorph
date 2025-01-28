import { createContext, Dispatch, SetStateAction } from "react";
import { Tool } from "./Tool";

export type CurrentTool = Tool | undefined;

interface CurrentToolContext {
  currentTool: CurrentTool;
  setCurrentTool: Dispatch<SetStateAction<CurrentTool>>;
}

export const CurrentToolContext = createContext<CurrentToolContext>({
  currentTool: undefined,
  setCurrentTool: () => {},
});
