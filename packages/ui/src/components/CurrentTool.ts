import { createContext, useContext, Dispatch, SetStateAction } from "react";
import { Tool } from "../tools/Tool";

export type CurrentTool = Tool | undefined;

export interface CurrentToolContext {
  currentTool: CurrentTool;
  setCurrentTool: Dispatch<SetStateAction<CurrentTool>>;
}

export const CurrentToolContext = createContext<CurrentToolContext>({
  currentTool: undefined,
  setCurrentTool: () => {},
});

export const useCurrentToolContext = (): CurrentToolContext => {
  return useContext(CurrentToolContext);
};
