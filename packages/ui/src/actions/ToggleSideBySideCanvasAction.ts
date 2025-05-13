import { TriggerAction } from "./Action";
import { KeyboardShortcut } from "./KeyboardShortcut";

import { DocumentManager } from "../doc/DocumentManager";
import { ViewContext } from "../view";

export class ToggleSideBySideCanvasAction extends TriggerAction {
  constructor() {
    super({
      name: "Toggle Side-by-Side Canvas",
      shortcut: new KeyboardShortcut("C"),
    });
  }

  onTrigger(_documentManager: DocumentManager, { view, setView }: ViewContext) {
    setView({ ...view, sideBySideCanvas: !view.sideBySideCanvas });
  }
}
