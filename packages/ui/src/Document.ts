import { Vector2 } from 'threejs-math';

/* Note: for now, we use the `A` prefix for classes meant to be used with
   Automerge, while both the old and new document definitions are in the
   repo.
*/

// TODO: decides whether to use "strong typing" for Document elements (e.g., with
// a `Layer` class, `Point` class, etc.), or if we simply go with lose
// typing, e.g., the same as the output of JSON.parse() (possibly defining a
// `Layer` interface, `Point` interface, etc.).

export class Point {
  constructor(
    public name: string = 'New Point',
    public position: Vector2 = new Vector2(0, 0)
  ) {}

  clone(): Point {
    return new Point().copy(this);
  }

  copy(source: Point): Point {
    this.name = source.name;
    this.position.copy(source.position);
    return this;
  }

  equals(other: Point): boolean {
    return this.name === other.name && this.position.equals(other.position);
  }
}

export interface APoint {
  name: string;
  position: Vector2;
}

/**
 * Stores information about a given layer.
 */
export class LayerProperties {
  constructor(public name: string = 'New Layer') {}

  /**
   * Returns a new LayerProperties with the same content as this one.
   */
  clone(): LayerProperties {
    return new LayerProperties().copy(this);
  }

  /**
   * Copies the content from the source layer into this one.
   */
  copy(source: LayerProperties): LayerProperties {
    this.name = source.name;
    return this;
  }

  equals(other: LayerProperties): boolean {
    return this.name === other.name;
  }
}

export interface ALayerProperties {
  name: string;
}

/**
 * Stores the data in a given Document layer.
 */
export class Layer {
  constructor(
    public properties: LayerProperties = new LayerProperties(),
    public points: Array<Point> = []
  ) {}

  /**
   * Returns a new layer with the same content as this one.
   */
  clone(): Layer {
    return new Layer().copy(this);
  }

  /**
   * Copies the content from the source layer into this one.
   */
  copy(source: Layer): Layer {
    this.properties = source.properties.clone();
    this.points = source.points.map(p => p.clone());
    return this;
  }

  /**
   * Adds a point to the layer.
   */
  addPoint(position: Vector2): Layer {
    const name = 'Point ' + (this.points.length + 1);
    this.points.push(new Point(name, position));
    return this;
  }
}

export interface ALayer {
  properties: ALayerProperties;
  points: Array<APoint>;
}

/**
 * Stores all objects in the document.
 */
export class Document {
  constructor(public layers: Array<Layer> = []) {}

  /**
   * Returns a new document with the same content as this one.
   */
  clone(): Document {
    return new Document().copy(this);
  }

  /**
   * Copies the content from the source document into this one.
   */
  copy(source: Document): Document {
    this.layers = source.layers.map(l => l.clone());
    return this;
  }

  /**
   * Adds a layer to the document at the given index.
   *
   * If `index` is -1 (the default), the layer is added last.
   */
  addLayer(index: number = -1): Document {
    const name = 'Layer ' + (this.layers.length + 1);
    const props = new LayerProperties(name);
    const layer = new Layer(props);
    if (index < 0) {
      this.layers.push(layer);
    } else {
      this.layers.splice(index, 0, layer);
    }
    return this;
  }

  /**
   * Removes the layer at the given index.
   */
  removeLayer(index: number): Document {
    this.layers.splice(index, 1);
    return this;
  }
}

export interface ADocument {
  layers: Array<ALayer>;
  activeLayerIndex: number /* XXX: move out of document? */;
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
  private _onChange: () => void;
  private _history: Array<Document>;
  private _index: number;
  private _workingCopy: Document;

  // TODO: better way to store which layer is active (e.g., some unique ID),
  // making it invariant to inserting/deleting layers or undoing such operations.
  private _activeLayerIndex: number;

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
  constructor(onChange?: () => void, history?: Array<Document>, index?: number, workingCopy?: Document) {
    this._version = 0;
    this._onChange = onChange !== undefined ? onChange : () => {};
    this._history = history !== undefined ? history : [new Document().addLayer()];
    this._index = index !== undefined ? index : this._history.length - 1;
    if (workingCopy !== undefined) {
      this._workingCopy = workingCopy;
    } else {
      // Same as `this._makeWorkingCopy();` but silences strictPropertyInitialization false positive
      this._workingCopy = this._history[this._index].clone();
    }
    this._activeLayerIndex = 0;
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

  private _notify(): void {
    this._version += 1;
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

  // TODO: move activeLayer to SelectionState class

  // -1 if no active layer
  activeLayerIndex(): number {
    const doc = this.document();
    if (!doc || doc.layers.length === 0) {
      return -1;
    }

    // clamp index to valid range, so that it
    // behaves nicely if the active layer is
    // the last layer and is deleted.
    let index = this._activeLayerIndex;
    if (index < 0) {
      index = 0;
    }
    if (index >= doc.layers.length) {
      index = doc.layers.length - 1;
    }
    return index;
  }

  activeLayer(): Layer | null {
    const doc = this.document();
    if (!doc) {
      return null;
    }
    const index = this.activeLayerIndex();
    if (0 <= index && index < doc.layers.length) {
      return doc.layers[index];
    } else {
      return null;
    }
  }

  setActiveLayer(index: number) {
    // Note: activeLayerIndex() and _activeLayerIndex might differ
    const prevIndex = this.activeLayerIndex();
    this._activeLayerIndex = index;
    if (this.activeLayerIndex() !== prevIndex) {
      this._notify();
    }
  }
}
