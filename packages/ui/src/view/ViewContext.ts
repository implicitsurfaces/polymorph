import { createContext, useContext, Dispatch, SetStateAction } from "react";
import { View, defaultView } from "./View";

export interface ViewContext {
  readonly view: View;
  readonly setView: Dispatch<SetStateAction<View>>;
}

export const ViewContext = createContext<ViewContext>({
  view: defaultView,
  setView: () => {},
});

export const useViewContext = (): ViewContext => {
  return useContext(ViewContext);
};
