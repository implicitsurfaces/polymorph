import { createContext, Dispatch, SetStateAction } from "react";

export type CurrentTool = string | undefined;

interface CurrentToolContext {
  currentTool: CurrentTool;
  setCurrentTool: Dispatch<SetStateAction<CurrentTool>>;
}

export const CurrentToolContext = createContext<CurrentToolContext>({
  currentTool: undefined,
  setCurrentTool: () => {},
});
