import { TriggerAction } from "./Action";
import { KeyboardShortcut } from "./KeyboardShortcut";

import { DocumentManager } from "../doc/DocumentManager";

export class OpenAction extends TriggerAction {
  constructor() {
    super({
      name: "Open",
      shortcut: new KeyboardShortcut("CtrlCmd+O"),
      menu: "",
      menuIndex: 3,
    });
  }

  onTrigger(documentManager: DocumentManager) {
    documentManager.open();
  }
}

export class SaveAction extends TriggerAction {
  constructor() {
    super({
      name: "Save",
      shortcut: new KeyboardShortcut("CtrlCmd+S"),
      menu: "",
      menuIndex: 4,
    });
  }

  onTrigger(documentManager: DocumentManager) {
    documentManager.save();
  }
}

export class SaveAsAction extends TriggerAction {
  constructor() {
    super({
      name: "Save As...",
      shortcut: new KeyboardShortcut("CtrlCmd+Shift+S"),
      menu: "",
      menuIndex: 5,
    });
  }

  onTrigger(documentManager: DocumentManager) {
    documentManager.saveAs();
  }
}
