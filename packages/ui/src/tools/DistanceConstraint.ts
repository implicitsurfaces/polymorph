import { ElementId, Element, Layer } from "../Document";
import { DocumentManager } from "../DocumentManager";
import { v4 as uuidv4 } from "uuid";

type ParamId = string;

type GeometricPrimitiveId = string;

interface GeometricPrimitiveBase {
  type: string;
  id: GeometricPrimitiveId;
}

interface Point extends GeometricPrimitiveBase {
  type: "Point";
  x: ParamId;
  y: ParamId;
}

type GeometricPrimitive = Point; // | Edge

/*
interface Edge extends GeometricPrimitive {
  startPoint: GeometryPrimitiveId;
  endPoint: GeometryPrimitiveId;
}
*/

type ConstraintId = string;

interface ConstraintBase {
  id: ConstraintId;
  type: string;
  equations: Array<string>;
  forwardSubs: Array<Array<string>>;
}

interface PointToPointDistanceConstraint extends ConstraintBase {
  type: "PointToPointDistanceConstraint";
  value: number;
  points: [GeometricPrimitiveId, GeometricPrimitiveId];
}

type Constraint = PointToPointDistanceConstraint; // | ...

type Geometry = Array<GeometricPrimitive>;
type Constraints = Array<Constraint>;
//type ParamValues = Map<ParamId, number>;

/*
function makeProgram(
  geometry: Geometry,
  constraints: Constraints,
  paramValues: ParamValues,
) {}
*/

const geometry: Geometry = [];
const constraints: Constraints = [];

function removeDashes(id: ElementId) {
  // id.replaceAll('-', ''); // Requires ES2021
  console.log(id);
  const res = id.replace(/-/g, "");
  console.log(res);
  return res;
}

export function addDistanceConstraint(documentManager: DocumentManager) {
  // Get the two selected points.
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const elements = selection.selectedElements();
  const errorMsg =
    "You need to select two points to add a distance constraint.";
  if (elements.length !== 2) {
    console.log(errorMsg);
    return;
  }
  const p1Id = elements[0];
  const p2Id = elements[0];
  const p1 = doc.getElementFromId<Element>(p1Id);
  const p2 = doc.getElementFromId<Element>(p2Id);
  if (p1?.type !== "Point" || p2?.type !== "Point") {
    console.log(errorMsg);
    return;
  }

  // Create geometry from scractch
  // TODO: persist it.
  //
  geometry.splice(0, geometry.length);
  for (const layerId of doc.layers) {
    const layer = doc.getElementFromId<Layer>(layerId);
    if (!layer) {
      continue;
    }
    for (const elementId of layer.elements) {
      const element = doc.getElementFromId<Element>(layerId);
      if (!element) {
        continue;
      }
      const id = removeDashes(elementId);
      if (element.type === "Point") {
        geometry.push({
          type: "Point",
          id: id,
          x: `p$(id)_x`,
          y: `p$(id)_y`,
        });
      }
    }
  }

  // TODO: current distance between selected points.
  const distanceConstraintValue = 100;

  constraints.push({
    type: "PointToPointDistanceConstraint",
    id: uuidv4(),
    equations: ["TODO"],
    forwardSubs: [],
    value: distanceConstraintValue,
    points: [p1.id, p2.id],
  });

  // geometry.solveConstraints(
  //   requestedValues,
  //   (constraintFn = null),
  //   (initValues = []),
  //   (deconstructed = null),
  // );

  console.log(elements.length);
}
