import { KeyboardShortcut } from "./KeyboardShortcut";
import { DocumentManager } from "../doc/DocumentManager";

export interface ActionProps {
  readonly name: string;
  readonly icon?: string;
  readonly shortcut?: KeyboardShortcut;
  readonly menu?: string;
  readonly menuIndex?: number;
}

export abstract class Action {
  /**
   * The human-readable name of the action, as appears for example in the
   * menu, status bar, or popup hints.
   */
  readonly name: string;

  /**
   * The icon that represents this action, if any. It appears for example in
   * the menu or tool button.
   */
  readonly icon?: string;

  /**
   * The shortcut that triggers this action, if any.
   */
  readonly shortcut?: KeyboardShortcut;

  /**
   * In which menu should this action appear, if any.
   *
   * Example:
   * - undefined: does not appear in any menu (this is the default)
   * - "": appears in the root menu
   * - "File": the file menu
   * - "Selection > Boolean Operations": the "Boolean Operation" submenu in the "Selection" menu.
   */
  readonly menu?: string;

  /**
   * A fractional index to specify at which position in the menu the action
   * should appear.
   */
  readonly menuIndex: number;

  constructor(props: ActionProps) {
    this.name = props.name;
    this.icon = props.icon;
    this.shortcut = props.shortcut;
    this.menu = props.menu;
    this.menuIndex = props.menuIndex ?? 0;
  }
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface TriggerActionProps extends ActionProps {}

export abstract class TriggerAction extends Action {
  constructor(props: TriggerActionProps) {
    super(props);
  }

  /**
   * This function is called when the user triggers the action, either by
   * clicking on the action's button or by typing its shortcut.
   *
   * This function should return `undefined` if the action was successful,
   * and otherwise return an information message intended for the user,
   * indicating why the action could not be performed(example: "Please select
   * two points.").
   */
  abstract onTrigger(documentManager: DocumentManager): string | void;
}
