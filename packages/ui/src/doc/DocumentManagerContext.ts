import { createContext } from "react";
import { DocumentManager } from "./DocumentManager";

interface DocumentManagerContext {
  documentManager: DocumentManager;
}

export const DocumentManagerContext = createContext<DocumentManagerContext>({
  documentManager: new DocumentManager(),
});
