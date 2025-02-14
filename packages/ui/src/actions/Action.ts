import { KeyboardShortcut } from "./KeyboardShortcut";
import { DocumentManager } from "../DocumentManager";

export interface Action {
  readonly name: string;
  readonly icon: string;
  readonly shortcut: KeyboardShortcut;

  /**
   * This function is called when the user triggers the action, either by
   * clicking on the action's button or by typing its shortcut.
   *
   * This function should return `undefined` if the action was successful,
   * and otherwise return an information message intended for the user,
   * indicating why the action could not be performed(example: "Please select
   * two points.").
   */
  readonly onTrigger?: (documentManager: DocumentManager) => string | undefined;
}
