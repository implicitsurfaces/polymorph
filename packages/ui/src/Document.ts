import { Vector2 } from "threejs-math";
import { v4 as uuidv4 } from "uuid";

export type ElementId = string;

interface Element {
  readonly id: ElementId;
  clone: () => Element;
}

/**
 * Any type that can be interpreted as a 2D vector, where the X and Y
 * coordinates are considered to be 0 if undefined.
 */
export type AnyVector2 = { x?: number; y?: number } | number[];

export function createVector2(data?: AnyVector2) {
  if (!data) {
    return new Vector2(0, 0);
  }
  let x = 0;
  let y = 0;
  if (Array.isArray(data)) {
    if (data.length > 0) {
      x = data[0];
    }
    if (data.length > 1) {
      y = data[1];
    }
  } else {
    if (data.x !== undefined) {
      x = data.x;
    }
    if (data.y !== undefined) {
      y = data.y;
    }
  }
  return new Vector2(x, y);
}

export interface AnyPointData {
  name?: string;
  position?: AnyVector2;
}

export class Point implements Element {
  public kind = "Point" as const;
  public name: string;
  public position: Vector2;

  static factory = (id: ElementId, data?: AnyPointData) => {
    return new Point(id, data);
  };

  constructor(
    readonly id: ElementId,
    data?: AnyPointData,
  ) {
    this.name = data?.name !== undefined ? data.name : "New Point";
    this.position = createVector2(data?.position);
  }

  clone(): Point {
    return new Point(this.id, this);
  }

  equals(other: Point): boolean {
    return this.name === other.name && this.position.equals(other.position);
  }
}

export class LayerProperties {
  constructor(public name: string = "New Layer") {}

  clone(): LayerProperties {
    return new LayerProperties(this.name);
  }
}

export interface AnyLayerData {
  properties?: LayerProperties;
  points?: Array<ElementId>;
}

export class Layer implements Element {
  public kind = "Layer" as const;
  public properties: LayerProperties;
  public points: Array<ElementId>;

  static factory = (id: ElementId, data?: AnyLayerData) => {
    return new Layer(id, data);
  };

  constructor(
    readonly id: ElementId,
    data?: AnyLayerData,
  ) {
    this.properties = data?.properties
      ? data.properties
      : new LayerProperties();
    this.points = data?.points ? data.points : [];
  }

  clone(): Layer {
    return new Layer(this.id, this);
  }
}

interface Clonable<T> {
  clone: () => T;
}

function cloneMap<ElementId, T extends Clonable<T>>(
  source: Map<ElementId, T>,
): Map<ElementId, T> {
  const dest = new Map<ElementId, T>();
  source.forEach((element, id) => {
    dest.set(id, element.clone());
  });
  return dest;
}

/**
 * Stores all objects in the document.
 */
export class Document {
  private _elements: Map<ElementId, Element>;

  public layers: Array<ElementId>;

  constructor(other?: Document) {
    if (other) {
      this._elements = cloneMap(other._elements);
      this.layers = [...other.layers];
    } else {
      this._elements = new Map();
      this.layers = [];
    }
  }

  /**
   * Returns a new document with the same content as this one.
   */
  clone(): Document {
    return new Document(this);
  }

  getElementFromId<T extends Element>(id: ElementId): T | undefined {
    const element: Element | undefined = this._elements.get(id);
    return element as T | undefined;
  }

  createElement<T extends Element, D>(
    factory: (id: ElementId, data: D) => T,
    data: D,
  ): T {
    const id = uuidv4();
    const element = factory(id, data);
    this._elements.set(id, element);
    return element;
  }

  createPoint(data: AnyPointData): Point {
    return this.createElement(Point.factory, data);
  }

  createLayer(data: AnyLayerData): Layer {
    return this.createElement(Layer.factory, data);
  }

  /**
   * Creates a new layer and add it to the document at the given index.
   *
   * If `index` is -1 (the default), the layer is added last.
   */
  addLayer(index: number = -1): Document {
    const name = `Layer ${this.layers.length + 1}`;
    const props = new LayerProperties(name);
    const layer = this.createLayer({ properties: props });
    if (index < 0) {
      this.layers.push(layer.id);
    } else {
      this.layers.splice(index, 0, layer.id);
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

  private _activeLayerId: ElementId;

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
    history?: Array<Document>,
    index?: number,
    workingCopy?: Document,
  ) {
    this._version = 0;
    this._onChange = onChange !== undefined ? onChange : () => {};
    this._history =
      history !== undefined ? history : [new Document().addLayer()];
    this._index = index !== undefined ? index : this._history.length - 1;
    if (workingCopy !== undefined) {
      this._workingCopy = workingCopy;
    } else {
      // Same as `this._makeWorkingCopy();` but silences strictPropertyInitialization false positive
      this._workingCopy = this._history[this._index].clone();
    }
    this._activeLayerId = this._workingCopy.layers[0];
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

  activeLayerId(): ElementId {
    return this._activeLayerId;
  }

  activeLayer(): Layer | undefined {
    const doc = this.document();
    if (!doc) {
      return undefined;
    }
    return doc.getElementFromId(this._activeLayerId);
  }

  setActiveLayer(id: ElementId) {
    if (this._activeLayerId !== id) {
      this._activeLayerId = id;
      this._notify();
    }
  }
}
