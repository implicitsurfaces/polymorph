import { createContext, useContext } from "react";
import { DocumentManager } from "./DocumentManager";

export const DocumentManagerContext = createContext<DocumentManager>(
  new DocumentManager(),
);

export const useDocumentManager = (): DocumentManager => {
  return useContext(DocumentManagerContext);
};
