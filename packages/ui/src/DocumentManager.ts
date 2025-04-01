import { Document, Layer, createTestDocument, Point } from "./Document";

import {
  ConstraintManager,
  RequestValues,
} from "./constraintSolving/ConstraintManager";

import { Selection } from "./Selection";

type SolveConstraintsOptions = {
  movedPoints?: Point[];
};

export interface ChangeNotificationOptions {
  readonly commit?: boolean; // default = true
  readonly buildConstraints?: boolean; // default = true
  readonly solveConstraints?: false | SolveConstraintsOptions; // default = {}
}

export class ChangeNotificationData {
  readonly commit: boolean;
  readonly buildConstraints: boolean;
  readonly solveConstraints: false | SolveConstraintsOptions;

  constructor(options: ChangeNotificationOptions) {
    this.commit = options.commit ?? true;
    this.buildConstraints = options.buildConstraints ?? false;
    this.solveConstraints = options.solveConstraints ?? {};
  }
}

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

  /**
   * Builds the constraint system. This should typically be called each time a
   * new constraint is added or the document changes.
   */
  private _buildConstraints() {
    this._constraintManager.updateConstraintFunction();
  }

  /**
   * Solves the constraint system.
   *
   * This mutates the document by updating geometric values (e.g., point
   * positions) such that the constraints are met.
   *
   * If given, `movedPoints` indicates a list of points that are explicitly
   * moved by the user, and therefore the solver will try to keep the current
   * position of these points unchanged.
   */
  // TODO: abstraction that generalizes "movedPoints" to other type of moved geometry?
  //
  private _solveConstraints(options: SolveConstraintsOptions) {
    const requested: RequestValues = [];
    if (options.movedPoints) {
      for (const point of options.movedPoints) {
        requested.push({
          ptId: point.id,
          axis: "x",
          param: point.x.id,
          value: point.x.value,
        });
        requested.push({
          ptId: point.id,
          axis: "y",
          param: point.y.id,
          value: point.y.value,
        });
      }
    }
    this._constraintManager.evaluateConstraintFunction({ requested });
  }

  /**
   * Notifies that the current document has changed, and therefore that any views
   * on the document must be update.
   *
   * If `options.commit` is `true` (the default is `true`), then the changes are
   * considered to be a complete undoable action, and a new entry in the
   * document history is created. This should typically be done for one-shot
   * actions (e.g., menu items) or on mouse release of a mouse drag action.
   *
   * If `options.commit` is `false`, then the changes are considered not to
   * represent a complete undoable action, and the document history is left
   * unchanged. This should typically be specified on mouse move of a mouse
   * drag action, since we do not want each individual mouse move to be
   * separately undoable.
   *
   * If `options.buildConstraints` is `true` (the default is `false`), then
   * the changes are considered to potentially modify the constraints, and
   * therefore the constraint system is (re-)built before being solved. This
   * should be set to `true` for changes affecting the constraint system,
   * that is, when there a locked measure is added or removed.
   *
   * If `options.solveConstraints` is not `false`` (the default is `{}`), then
   * the changes are considered to potentially violate the constraints, and
   * therefore the constraint system is (re-)solved, potentially modifying
   * the document. This solve step is always performed before creating the
   * entry in the document history (if any), in order to only store in the
   * history the version of the document with constraints resolved.
   */
  notifyChanges(options?: ChangeNotificationOptions): void {
    const data = new ChangeNotificationData(options ?? {});
    if (data.buildConstraints) {
      this._buildConstraints();
    }
    if (data.solveConstraints) {
      // Solve for the constraints (= locked measures) and update other
      // measure values (= unlocked measures).
      this._solveConstraints(data.solveConstraints);
    }
    if (data.commit) {
      // Wipe out any pre-existing redoable data
      this._history.splice(this.index() + 1);

      // Insert the current working copy as last element in the history.
      this._history.push(this._workingCopy);

      // Set it as new current index.
      this._index = this._history.length - 1;
      this._makeWorkingCopy();
    }

    // Update external document views (canvas, panels, etc.)
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
