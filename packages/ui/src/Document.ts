import { Vector2 } from 'threejs-math';

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
}

/**
 * Stores all objects in the document.
 */
export class Document {
  constructor(public points: Array<Point> = []) {}

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
    this.points = source.points.map(p => p.clone());
    return this;
  }

  /**
   * Adds a point to the document.
   */
  addPoint(position: Vector2): Document {
    const name = 'Point ' + (this.points.length + 1);
    this.points.push(new Point(name, position));
    return this;
  }
}

// https://stackoverflow.com/questions/424292/seedable-javascript-random-number-generator
class PseudoRandomNumberGenerator {
  private m;
  private a;
  private c;
  private state;

  constructor(seed?: number) {
    // LCG using GCC's constants
    this.m = 0x80000000; // 2**31;
    this.a = 1103515245;
    this.c = 12345;
    this.state = seed !== undefined ? seed : Math.floor(Math.random() * (this.m - 1));
  }

  nextInt() {
    this.state = (this.a * this.state + this.c) % this.m;
    return this.state;
  }

  nextRange(start: number, end: number) {
    // returns in range [start, end): including start, excluding end
    // can't modulu nextInt because of weak randomness in lower bits
    var rangeSize = end - start;
    var randomUnder1 = this.nextInt() / this.m;
    return start + Math.floor(randomUnder1 * rangeSize);
  }
}

export function testDocument(): Document {
  const prng = new PseudoRandomNumberGenerator(42);
  const document = new Document();
  const numPoints = 1000;
  const documentSize = 500;
  for (let i = 0; i < numPoints; ++i) {
    const x = prng.nextRange(-documentSize, documentSize);
    const y = prng.nextRange(-documentSize, documentSize);
    document.addPoint(new Vector2(x, y));
  }
  return document;
}

/**
 * Stores and manages the undo-redo history of the document.
 *
 * The idea is to avoid cloning the whole document at each
 * mouse move when the user is modifying an object's
 * property via a mouse drag action.
 */
export class DocumentManager {
  private _onChange;
  private _history: Array<Document>;
  private _index: number;
  private _workingCopy: Document;

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
    this._onChange = onChange !== undefined ? onChange : () => {};
    this._history = history !== undefined ? history : [testDocument()];
    this._index = index !== undefined ? index : this._history.length - 1;
    if (workingCopy !== undefined) {
      this._workingCopy = workingCopy;
    } else {
      // Same as `this._makeWorkingCopy();` but silences strictPropertyInitialization false positive
      this._workingCopy = this._history[this._index].clone();
    }
  }

  /**
   * Returns a shallow copy of this DocumentManager.
   */
  shallowClone(): DocumentManager {
    return new DocumentManager(this._onChange, this._history, this._index, this._workingCopy);
  }

  /**
   * Update what happens on change.
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
    this._onChange();

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
    this._onChange();
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
    this._onChange();
  }

  /**
   * Notifies that the current document may have changed, but that we want to
   * discard these changes and go back to the latest commited changes.
   */
  discardChanges(): void {
    this._makeWorkingCopy();
    this._onChange();
  }
}
