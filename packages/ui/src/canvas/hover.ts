import { Vector2 } from "threejs-math";
import { Camera2 } from "./Camera2.ts";
import { Selectable } from "../Selection.ts";
import { DocumentManager } from "../DocumentManager.ts";
import {
  Document,
  ElementId,
  isEdgeElement,
  Layer,
  Point,
} from "../Document.ts";
import {
  ControlPoint,
  getEdgeShapesAndControls,
  CanvasShape,
  CanvasArc,
  CanvasLineSegment,
} from "./drawEdges.ts";
import { controlPointRadius, edgeWidth, pointRadius } from "./style.ts";

import { CanvasPointerEvent } from "../canvas/events";

interface ClosestSelectable {
  selectable: Selectable | undefined;
  distance: number;
}

type DistanceFunction<T> = (element: T, position: Vector2) => number;
type MakeSelectableFunction<T> = (element: T, id: ElementId) => Selectable;
type Filter = (id: ElementId) => boolean;

function distToPoint(point: Point, position: Vector2): number {
  return point.position.distanceTo(position);
}

function selectPoint(point: Point): Selectable {
  return { type: "Element", id: point.id };
}

function distToControlPoint(cp: ControlPoint, position: Vector2): number {
  return cp.position.distanceTo(position);
}

function selectControlPoint(cp: ControlPoint, id: ElementId): Selectable {
  return { type: "SubElement", id: id, subName: cp.name };
}

function distToLineSegment(seg: CanvasLineSegment, position: Vector2): number {
  const a = seg.startPoint;
  const b = seg.endPoint;
  const segDir = b.clone().sub(a);
  const l = segDir.length();
  if (l > 0) {
    // Compute projection to the line segment
    segDir.divideScalar(l);
    const ap = position.clone().sub(a);
    const t = ap.dot(segDir);
    if (t < 0) {
      // Case where the projection is before startPoint
      return position.distanceTo(a);
    } else if (t > l) {
      // Case where the projection is after endPoint
      return position.distanceTo(b);
    } else {
      // Case where the projection is contained in the line segment
      return Math.abs(ap.cross(segDir));
    }
  } else {
    // Special case where the line segment is reduced to a point
    return position.distanceTo(a);
  }
}

function getArcPoint(arc: CanvasArc, angle: number): Vector2 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return new Vector2(c * arc.radius, s * arc.radius).add(arc.center);
}

// The distance from a point P to an arc is realized either at the arc end
// points, or at an interior arc point Q such that PQ is normal to the arc.
//
// There are potentially two such interior points Q, which are at the
// intersection between PC (C = circle center) and the circle itself:
// - the circle point closest to P (if this point belongs to the arc)
// - the circle point furthest to P (if this point belongs to the arc)
//
// We can ignore the latter, since by definition it is further from P than the
// arc end points.
//
// If the former does belong to the arc, then it directly gives the shortest
// distance. Otherwise, the distance is the min between the two arc end points.

function distToArc(arc: CanvasArc, position: Vector2): number {
  const dir = position.clone().sub(arc.center);
  let startAngle = arc.startAngle; // in [0, 2pi)
  let endAngle = arc.endAngle; // in [0, 2pi)
  let angle = dir.angle(); // in [0, 2pi)

  // Represent the arc as an increasing number interval, s.t.:
  // - endAngle is in [startAngle, startAngle + 2pi)
  // - angle1 is in [startAngle, startAngle + 2pi)
  // - angle2 is in [startAngle, startAngle + 2pi)
  if (arc.isCounterClockwise) {
    startAngle = arc.endAngle;
    endAngle = arc.startAngle;
  }
  if (endAngle < startAngle) {
    endAngle += 2 * Math.PI;
  }
  if (angle < startAngle) {
    angle += 2 * Math.PI;
  }

  if (angle <= endAngle) {
    // Case where angle is within the arc
    return Math.abs(dir.length() - arc.radius);
  } else {
    // Case then angle is not within the arc
    const d1 = getArcPoint(arc, startAngle).distanceTo(position);
    const d2 = getArcPoint(arc, endAngle).distanceTo(position);
    return Math.min(d1, d2);
  }
}

function distToShape(shape: CanvasShape, position: Vector2): number {
  switch (shape.type) {
    case "Arc":
      return distToArc(shape, position);
    case "LineSegment":
      return distToLineSegment(shape, position);
  }
}

function distToShapes(shapes: Array<CanvasShape>, position: Vector2): number {
  return shapes.reduce(
    (dist, shape) => Math.min(dist, distToShape(shape, position)),
    Infinity,
  );
}

function selectEdge(_shapes: Array<CanvasShape>, id: ElementId): Selectable {
  return { type: "Element", id: id };
}

function findClosestSelectableInLayer(
  doc: Document,
  camera: Camera2,
  layer: Layer,
  position: Vector2,
  filter?: Filter,
): ClosestSelectable {
  //
  const csPoint: ClosestSelectable = {
    selectable: undefined,
    distance: Infinity,
  };
  const csEdge: ClosestSelectable = {
    selectable: undefined,
    distance: Infinity,
  };

  function update<T>(
    cs: ClosestSelectable,
    element: T,
    dist: DistanceFunction<T>,
    select: MakeSelectableFunction<T>,
    id: ElementId,
  ) {
    const d = dist(element, position);
    if (d < cs.distance) {
      cs.selectable = select(element, id);
      cs.distance = d;
    }
  }

  for (const id of layer.elements) {
    const element = doc.getElementFromId(id);
    if (element && (!filter || filter(id))) {
      if (element.type === "Point") {
        update(csPoint, element, distToPoint, selectPoint, id);
      } else if (isEdgeElement(element)) {
        // TODO: cache the controls from the draw call?
        const sc = getEdgeShapesAndControls(doc, element);
        for (const cp of sc.controlPoints) {
          update(csPoint, cp, distToControlPoint, selectControlPoint, id);
        }
        update(csEdge, sc.shapes, distToShapes, selectEdge, id);
      }
    }
  }

  // Convert "distance to center" to "distance to disk".
  //
  // This is important in order to get a distance of exactly zero when the
  // mouse is inside the point's disk, in which case we want to select the
  // point even if the mouse is closer to some edge centerline than to the
  // point's center.
  //
  // In case the mouse is inside two or more point disks, we want to select
  // the point whose center is closest to the mouse, which is why the code
  // below must be a post-process step, and not part of the distance function
  // used above.
  //
  if (csPoint.selectable) {
    let radius = pointRadius;
    if (csPoint.selectable.type === "SubElement") {
      radius = controlPointRadius;
    }
    radius /= camera.zoom;
    if (csPoint.distance <= radius) {
      csPoint.distance = 0;
    } else {
      csPoint.distance -= radius;
    }
  }

  // Convert "distance to centerline" to "distance to stroke".
  //
  if (csEdge.selectable) {
    const hw = (0.5 * edgeWidth) / camera.zoom;
    if (csEdge.distance <= hw) {
      csEdge.distance = 0;
    } else {
      csEdge.distance -= hw;
    }
  }

  // Returns the found Point or Edge, whichever is closest,
  // with priority to the Point over the Edge in case of tie.
  if (csPoint.selectable) {
    if (csEdge.selectable) {
      return csPoint.distance <= csEdge.distance ? csPoint : csEdge;
    } else {
      return csPoint;
    }
  } else {
    return csEdge;
  }
}

function findClosestSelectableInDocument(
  doc: Document,
  camera: Camera2,
  position: Vector2,
  filter?: Filter,
): ClosestSelectable {
  let closestDistance = Infinity;
  let selectable: Selectable | undefined = undefined;
  for (const id of doc.layers) {
    const layer = doc.getElementFromId<Layer>(id);
    if (layer) {
      const ce = findClosestSelectableInLayer(
        doc,
        camera,
        layer,
        position,
        filter,
      );
      if (ce.distance < closestDistance) {
        closestDistance = ce.distance;
        selectable = ce.selectable;
      }
    }
  }
  return { selectable: selectable, distance: closestDistance };
}

// Note: For now, the hover function is in the canvas folder because there is
// indeed a semantic dependency on the canvas. For example, it has to know
// how the canvas renders the elements (e.g., the radius of a Point) in order
// to determine whether they are hovered. In particular, this is why it needs
// the `camera` parameter, since points are drawn with a fixed radius in
// screen space. In the future, we might want to abstract this away in some
// RenderSettings class, so that the hover function depends on
// RenderSettings, instead of the current implicit coupling with the draw
// code of the Canvas.
//
export function hover(
  documentManager: DocumentManager,
  camera: Camera2,
  mousePosition: Vector2,
  tolerance?: number,
  filter?: Filter,
) {
  if (tolerance === undefined) {
    const toleranceInPx = 3;
    tolerance = toleranceInPx / camera.zoom;
  }
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const ce = findClosestSelectableInDocument(
    doc,
    camera,
    mousePosition,
    filter,
  );
  if (ce.selectable && ce.distance < tolerance) {
    selection.setHovered(ce.selectable);
  } else {
    selection.setHovered(undefined);
  }
}

export function hoverFromCanvas(event: CanvasPointerEvent) {
  hover(event.documentManager, event.camera, event.documentPosition);
}
