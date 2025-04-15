import { TriggerAction } from "./Action";
import { KeyboardShortcut } from "./KeyboardShortcut";

import { DocumentManager } from "../doc/DocumentManager";

export class UndoAction extends TriggerAction {
  constructor() {
    super({
      name: "Undo",
      shortcut: new KeyboardShortcut("CtrlCmd+Z"),
      menu: "",
      menuIndex: 1,
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
      menu: "",
      menuIndex: 2,
    });
  }

  onTrigger(documentManager: DocumentManager) {
    documentManager.redo();
  }
}
