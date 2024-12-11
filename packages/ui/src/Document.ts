import { Vector2 } from "threejs-math";
import { v4 as uuidv4 } from "uuid";

///////////////////////////////////////////////////////////////////////////////
//                             Base types

export type ElementId = string;

export interface ElementBaseOptions {
  name?: string;
}

export interface ElementBaseData {
  name: string;
}

export interface ElementBase extends ElementBaseData {
  readonly id: ElementId;
  readonly type: string;
}

export interface ElementSpec<T, Options> {
  create: (id: ElementId, options: Options) => T;
  clone: (other: T) => T;
}

// Note: we currently need clone() for deep cloning purposes.
//
// If all data properties of elements were immutable types (which is currently
// not the case, e.g., Vector2 and Array<ElementId>), then we could instead
// do a shallow copy via `clonedElement = {...element}` which would make
// element types more convenient to implement, and would likely improve
// performance.

///////////////////////////////////////////////////////////////////////////////
//                               Point

export interface PointOptions extends ElementBaseOptions {
  position?: Vector2;
}

export interface PointData extends ElementBaseData {
  position: Vector2;
}

export interface Point extends ElementBase, PointData {
  type: "Point";
}

export const PointDefaultOptions = {
  name: "Point",
  position: new Vector2(0, 0),
};

export const Point: ElementSpec<Point, PointOptions> = {
  create: (id: ElementId, options: PointOptions) => {
    return {
      id: id,
      type: "Point",
      ...PointDefaultOptions,
      ...options,
    };
  },
  clone: (other: Point) => {
    return { ...other, position: other.position.clone() };
  },
};

///////////////////////////////////////////////////////////////////////////////
//                               Layer

export interface LayerOptions extends ElementBaseOptions {
  points?: Array<ElementId>;
}

export interface LayerData extends ElementBaseData {
  points: Array<ElementId>;
}

export interface Layer extends ElementBase, LayerData {
  type: "Layer";
}

export const LayerDefaultOptions = {
  name: "Layer",
  points: [],
};

export const Layer: ElementSpec<Layer, LayerOptions> = {
  create: (id: ElementId, options: LayerOptions) => {
    return {
      id: id,
      type: "Layer",
      ...LayerDefaultOptions,
      ...options,
    };
  },
  clone: (other: Layer) => {
    return { ...other, points: [...other.points] };
  },
};

///////////////////////////////////////////////////////////////////////////////
//                               Tagged Union

export type Element = Point | Layer;

///////////////////////////////////////////////////////////////////////////////
//                               Util

function cloneElement(element: Element) {
  // Note: this could be simplified to: `return {.. element}` if all element
  // properties where immutable.
  switch (element.type) {
    case "Point":
      return Point.clone(element);
    case "Layer":
      return Layer.clone(element);
  }
}

function cloneElementMap(source: Map<ElementId, Element>) {
  const dest = new Map<ElementId, Element>();
  source.forEach((element, id) => {
    dest.set(id, cloneElement(element));
  });
  return dest;
}

///////////////////////////////////////////////////////////////////////////////
//                               Document

/**
 * Stores all objects in the document.
 */
export class Document {
  private _elements: Map<ElementId, Element>;

  public layers: Array<ElementId>;

  constructor(other?: Document) {
    if (other) {
      this._elements = cloneElementMap(other._elements);
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

  createElement<T extends Element, Options>(
    spec: ElementSpec<T, Options>,
    options: Options,
  ): T {
    const id = uuidv4();
    const element = spec.create(id, options);
    this._elements.set(id, element);
    return element;
  }

  createPoint(options: PointOptions): Point {
    return this.createElement(Point, options);
  }

  createLayer(options: LayerOptions): Layer {
    return this.createElement(Layer, options);
  }

  /**
   * Creates a new layer and add it to the document at the given index.
   *
   * If `index` is -1 (the default), the layer is added last.
   */
  createLayerAtIndex(index: number = -1): Document {
    const name = `Layer ${this.layers.length + 1}`;
    const layer = this.createLayer({ name: name });
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
