import { Vector2 } from "threejs-math";
import { v4 as uuidv4 } from "uuid";

export type ElementId = string;

export interface Element {
  readonly id: ElementId;
  readonly type: string;
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
  public type = "Point" as const;
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

  equals(other: LayerProperties): boolean {
    return this.name === other.name;
  }
}

export interface AnyLayerData {
  properties?: LayerProperties;
  points?: Array<ElementId>;
}

export class Layer implements Element {
  public type = "Layer" as const;
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
  createLayerAtIndex(index: number = -1): Document {
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
  deleteLayerAtIndex(index: number): Document {
    const id = this.layers[index];
    if (id === undefined) {
      return this;
    }
    const layer = this.getElementFromId<Layer>(id);
    if (layer === undefined) {
      return this;
    }
    this.layers.splice(index, 1);
    this._elements.delete(id);
    return this;
  }
}
