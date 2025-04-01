import { Document, Layer, createTestDocument, Point } from "./Document";

import {
  ConstraintManager,
  RequestValues,
} from "./constraintSolving/ConstraintManager";

import { Selection } from "./Selection";

type eventTypeString =
  | "MOVE"
  | "END_MOVE"
  | "START_LINE"
  | "PLACE_LINE"
  | "END_LINE"
  | "CREATE_LAYER"
  | "DELETE_LAYER"
  | "SET_POINT"
  | "ADD_CONSTRAINT"
  | "PLACE_POINT"
  | "CHANGED_LOCK";

/**
 * Stores and manages the undo-redo history of the document.
 *
 * The idea is to avoid cloning the whole document at each
 * mouse move when the user is modifying an object's
 * property via a mouse drag action.
 */
export class DocumentManager {
  private _version: number;

  private _constraintManager: ConstraintManager;

  private _onChange: () => void;
  private _history: Document[];
  private _index: number;
  private _workingCopy: Document;
  private _selection: Selection;

  /**
   * Constructs a new `DocumentManager`.
   *
   * The given `onChange` callback will be called whenever a change to the
   * current document is notified via the `stageChanges()` or `commitChanges()`
   * methods.
   *
   * For example, in the context of a React application, the `onChange()` callback
   * can be creating a `shallowClone()` of the `DocumentManager` and assigning it as
   * new state, so that React knows that the component (and its subcomponents if the
   * `DocumentManager` is passed as prop) should be re-rendered.
   */
  constructor(
    onChange?: () => void,
    history?: Document[],
    index?: number,
    workingCopy?: Document,
  ) {
    this._version = 0;
    this._onChange = onChange !== undefined ? onChange : () => {};
    this._history = history !== undefined ? history : [createTestDocument()];
    this._index = index !== undefined ? index : this._history.length - 1;
    if (workingCopy !== undefined) {
      this._workingCopy = workingCopy;
    } else {
      // Same as `this._makeWorkingCopy();` but silences strictPropertyInitialization false positive
      this._workingCopy = this._history[this._index].clone();
    }
    this._selection = new Selection(() => {
      this._notify();
    });
    this._ensureActiveLayer();
    this._constraintManager = new ConstraintManager(this._workingCopy);
  }

  /**
   * Returns the current version. This is a number that starts at 0 and is
   * incremented whenever (and just before) the `onChange()` callback is
   * called.
   *
   * You can use this version in dependency arrays to cause rerenders or
   * re-evaluate callbacks/effects.
   */
  version(): number {
    return this._version;
  }

  private _ensureActiveLayer() {
    const doc = this.document();
    const selection = this.selection();
    if (doc.layers.length > 0) {
      if (!doc.getNode(selection.activeLayerId(), Layer)) {
        selection.setActiveLayerId(doc.layers[0]);
      }
    }
  }

  private _notify(): void {
    this._version += 1;
    this._ensureActiveLayer();
    this._onChange();
  }

  /**
   * Provides a callback that is called each time the `version()` changes.
   */
  onChange(fn: () => void): void {
    this._onChange = fn;
  }

  /**
   * Returns the current document.
   */
  document(): Document {
    return this._workingCopy;
  }

  /**
   * Returns the current history index of the document.
   */
  index(): number {
    return this._index;
  }

  // Sets the working copy to be a deep copy of the given history index.
  //
  private _makeWorkingCopy(): void {
    this._workingCopy = this._history[this._index].clone();
    this._constraintManager.setDocument(this._workingCopy);
  }

  /**
   * Make the given history `index` the current document.
   */
  goToIndex(index: number): void {
    const min = 0;
    const max = this._history.length - 1;
    const clampedIndex = Math.min(Math.max(index, min), max);

    this._index = clampedIndex;
    this._makeWorkingCopy();
    this._notify();

    // Note: if the current index is already equal to clampedIndex,
    // we still need to make a new working copy since there may be
    // changes (staged or not) in the document which we need to discard,
    // and the current implementation is not aware of unstaged changes.
  }

  /**
   * Move back one step in history.
   */
  undo(): void {
    this.goToIndex(this.index() - 1);
  }

  /**
   * Move forward one step in history.
   */
  redo(): void {
    this.goToIndex(this.index() + 1);
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  dispatchEvent(eventType: eventTypeString, data: any = {}): void {
    // console.log("EVENT_TYPE:", eventType);

    switch (eventType) {
      case "MOVE": {
        const requested: RequestValues = [];

        data.movedPoints.forEach((pt: Point) => {
          requested.push({
            ptId: pt.id,
            axis: "x",
            param: pt.x.id,
            value: pt.x.value,
          });

          requested.push({
            ptId: pt.id,
            axis: "y",
            param: pt.y.id,
            value: pt.y.value,
          });
        });

        // solve constraints
        this._constraintManager.evaluateConstraintFunction({ requested });
        this.stageChanges();
        break;
      }
      case "END_MOVE":
        // solve constraints
        this._constraintManager.evaluateConstraintFunction();
        this.commitChanges();
        break;
      case "START_LINE":
        this.stageChanges();
        break;
      case "PLACE_LINE":
        this.stageChanges();
        break;
      case "END_LINE":
        this.commitChanges();
        break;
      case "CREATE_LAYER":
        this.commitChanges();
        break;
      case "DELETE_LAYER":
        this.commitChanges();
        break;
      case "SET_POINT":
        // set constraint program values
        this._constraintManager.evaluateConstraintFunction();
        this.commitChanges();
        break;
      case "ADD_CONSTRAINT":
        // analyze constraint system
        this._constraintManager.updateConstraintFunction();
        this._constraintManager.evaluateConstraintFunction();
        this.commitChanges();
        break;
      case "PLACE_POINT":
        this.commitChanges();
        break;
      case "CHANGED_LOCK":
        this.commitChanges();
        break;
      default:
        console.log("Unknown event:", eventType);
        break;
    }
  }

  /**
   * Notifies that the current document has changed (and therefore that any views
   * on the document must be updated), but that these changes do not yet
   * represent a complete undoable action.
   *
   * Typically, this should be called on mouse move of a mouse drag action.
   */
  stageChanges(): void {
    this._notify();
  }

  /**
   * Notifies that the current document has changed (and therefore that any views
   * on the document must be updated), and that these changes represent a
   * complete undoable action, therefore creating a new entry in the document
   * history.
   *
   * Typically, this should be called for one-shot actions (e.g., menu items)
   * or on mouse release of a mouse drag action.
   */
  // TODO: commit message to show in History Panel?
  commitChanges(): void {
    // Wipe out any pre-existing redoable data
    this._history.splice(this.index() + 1);

    // Insert the current working copy as last element in the history.
    this._history.push(this._workingCopy);

    // Set it as new current index.
    this._index = this._history.length - 1;
    this._makeWorkingCopy();

    // Notify
    this._notify();
  }

  /**
   * Notifies that the current document may have changed, but that we want to
   * discard these changes and go back to the latest commited changes.
   */
  discardChanges(): void {
    this._makeWorkingCopy();
    this._notify();
  }

  selection(): Selection {
    return this._selection;
  }
}
