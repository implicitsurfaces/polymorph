import { TriggerAction } from "./Action";
import { KeyboardShortcut } from "./KeyboardShortcut";

import { DocumentManager } from "../DocumentManager";

export class UndoAction extends TriggerAction {
  constructor() {
    super({
      name: "Undo",
      shortcut: new KeyboardShortcut("CtrlCmd+Z"),
    });
  }

  onTrigger(documentManager: DocumentManager) {
    documentManager.undo();
  }
}

export class RedoAction extends TriggerAction {
  constructor() {
    super({
      name: "Redo",
      shortcut: new KeyboardShortcut("CtrlCmd+Shift+Z"),
    });
  }

  onTrigger(documentManager: DocumentManager) {
    documentManager.redo();
  }
}
