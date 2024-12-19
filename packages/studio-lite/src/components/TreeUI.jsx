import { useEffect } from "react";
import { TreeRepr } from "./TreeRepr";
import { observer } from "mobx-react";
import useEditorStore from "../state/useEditorStore";
import LoadingScreen from "./LoadingScreen";

export const TreeUI = observer(function TreeUI() {
  const store = useEditorStore();

  useEffect(() => {
    store.computeTreeReprs();
    return () => {
      store.clearTreeReprs();
    };
  }, [store]);

  if (!store.currentTreeReprs) {
    return <LoadingScreen />;
  }

  return <TreeRepr tree={store.currentTreeReprs.image} />;
});
