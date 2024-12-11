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
  readonly create: (id: ElementId, options: Options) => T;
  readonly clone: (other: T) => T;
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
//                               EdgeBase

export interface EdgeBaseOptions extends ElementBaseOptions {
  startPoint?: ElementId;
  endPoint?: ElementId;
}

export interface EdgeBaseData extends ElementBaseData {
  startPoint: ElementId;
  endPoint: ElementId;
}

///////////////////////////////////////////////////////////////////////////////
//                               LineSegment

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface LineSegmentOptions extends EdgeBaseOptions {
  // No additional options w.r.t EdgeBase
}

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface LineSegmentData extends EdgeBaseData {
  // No additional data w.r.t EdgeBase
}

export interface LineSegment extends ElementBase, LineSegmentData {
  type: "LineSegment";
}

export const LineSegmentDefaultOptions = {
  name: "Line Segment",
  startPoint: "",
  endPoint: "",
};

export const LineSegment: ElementSpec<LineSegment, LineSegmentOptions> = {
  create: (id: ElementId, options: LineSegmentOptions) => {
    return {
      id: id,
      type: "LineSegment",
      ...LineSegmentDefaultOptions,
      ...options,
    };
  },
  clone: (other: LineSegment) => {
    return { ...other };
  },
};

///////////////////////////////////////////////////////////////////////////////
//                               Layer

export interface LayerOptions extends ElementBaseOptions {
  elements?: Array<ElementId>;
}

export interface LayerData extends ElementBaseData {
  elements: Array<ElementId>;
}

export interface Layer extends ElementBase, LayerData {
  type: "Layer";
}

export const LayerDefaultOptions = {
  name: "Layer",
  elements: [],
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
    return { ...other, elements: [...other.elements] };
  },
};

///////////////////////////////////////////////////////////////////////////////
//                               Tagged Union

export type EdgeElement = LineSegment; // | Arc | SCurve | ...
export type SkeletonElement = Point | EdgeElement;

export type Element = SkeletonElement | Layer;

///////////////////////////////////////////////////////////////////////////////
//                               Util

function cloneElement(element: Element): Element {
  // Note: this could be simplified to: `return {.. element}` if all element
  // properties where immutable.
  switch (element.type) {
    case "Point":
      return Point.clone(element);
    case "Layer":
      return Layer.clone(element);
    case "LineSegment":
      return LineSegment.clone(element);
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

  /**
   * Creates a new layer and add it to the document at the given index.
   *
   * If `index` is -1 (the default), the layer is added last.
   */
  createLayerAtIndex(index: number = -1): Document {
    const name = `Layer ${this.layers.length + 1}`;
    const layer = this.createElement(Layer, { name: name });
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

///////////////////////////////////////////////////////////////////////////////
//                            Test Document

export function createTestDocument() {
  const doc = new Document();
  const layer = doc.createElement(Layer, { name: "Layer 1" });
  doc.layers = [layer.id];
  const p1 = doc.createElement(Point, {
    name: "Point 1",
    position: new Vector2(0, 0),
  });
  const p2 = doc.createElement(Point, {
    name: "Point 2",
    position: new Vector2(100, 0),
  });
  const p3 = doc.createElement(Point, {
    name: "Point 3",
    position: new Vector2(100, 100),
  });
  const s1 = doc.createElement(LineSegment, {
    name: "Segment 1",
    startPoint: p1.id,
    endPoint: p2.id,
  });
  const s2 = doc.createElement(LineSegment, {
    name: "Segment 2",
    startPoint: p2.id,
    endPoint: p3.id,
  });
  layer.elements = [p1.id, p2.id, p3.id, s1.id, s2.id];
  return doc;
}
