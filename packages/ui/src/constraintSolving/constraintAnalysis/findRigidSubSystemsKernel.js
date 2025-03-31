import { getKernel } from "./getKernel.js";

const EPSILON = 1e-8;

export function findRigidSubSystemsKernel({
  ptIdVec,
  ptValVec,
  jacobian,
  coincident,
  points,
}) {
  if (jacobian.length === 0) return [];

  const kernel = getKernel(jacobian, EPSILON);

  roundMatrix(kernel, EPSILON);

  // Map points to their representative in coincident groups
  const pointRepresentative = {};
  if (coincident && coincident.length) {
    for (const group of coincident) {
      for (const pointId of group) {
        pointRepresentative[pointId] = group[0]; // Use first point as representative
      }
    }
  }

  const idXY = [];
  for (let i = 0; i < points.length; i += 1) {
    const id = points[i].id;
    const x = ptValVec[i * 2];
    const y = ptValVec[i * 2 + 1];
    idXY.push({ id, x, y });
  }

  const graph = {};
  for (let p of idXY) {
    graph[p.id] = new Set();
  }

  const allRigidPairs = [];
  for (let i = 0; i < idXY.length; i++) {
    for (let j = i + 1; j < idXY.length; j++) {
      const pt0 = idXY[i];
      const pt1 = idXY[j];

      // Skip if points are coincident
      const rep0 = pointRepresentative[pt0.id] || pt0.id;
      const rep1 = pointRepresentative[pt1.id] || pt1.id;
      if (rep0 === rep1) {
        continue;
      }

      const grad = getDistanceGradient(i, j, ptValVec);
      const inColSpace = isInColumnSpace(grad, kernel, EPSILON);
      if (inColSpace) {
        graph[pt0.id].add(pt1.id);
        graph[pt1.id].add(pt0.id);
        allRigidPairs.push([pt0.id, pt1.id]);
      }
    }
  }

  // Apply coincident relationships to the graph
  if (coincident && coincident.length) {
    for (const group of coincident) {
      for (let i = 0; i < group.length; i++) {
        for (let j = i + 1; j < group.length; j++) {
          if (graph[group[i]] && graph[group[j]]) {
            // Make coincident points share the same connections
            const unionNeighbors = new Set([
              ...graph[group[i]],
              ...graph[group[j]],
            ]);
            graph[group[i]] = unionNeighbors;
            graph[group[j]] = unionNeighbors;
          }
        }
      }
    }
  }

  const rigidSubSystems = [];
  const visited = new Set();

  for (let i = 0; i < idXY.length; i++) {
    const startPointId = idXY[i].id;
    if (visited.has(startPointId)) continue;

    const neighbors = Array.from(graph[startPointId]);
    for (let j = 0; j < neighbors.length; j++) {
      const secondPointId = neighbors[j];
      if (visited.has(secondPointId)) continue;

      const rigidSystem = new Set([startPointId, secondPointId]);
      let systemGrew = true;

      while (systemGrew) {
        systemGrew = false;
        for (let k = 0; k < idXY.length; k++) {
          const candidateId = idXY[k].id;
          if (rigidSystem.has(candidateId)) continue;

          let connections = 0;
          for (const existingId of rigidSystem) {
            if (graph[candidateId].has(existingId)) {
              connections++;
            }
            if (connections >= 2) {
              rigidSystem.add(candidateId);
              systemGrew = true;
              break;
            }
          }
        }
      }

      if (rigidSystem.size > 2) {
        // Include all coincident points
        const expandedSystem = new Set(rigidSystem);
        if (coincident && coincident.length) {
          for (const group of coincident) {
            if (group.some((id) => rigidSystem.has(id))) {
              for (const id of group) {
                expandedSystem.add(id);
              }
            }
          }
        }

        rigidSubSystems.push(Array.from(expandedSystem));
        for (const id of expandedSystem) {
          visited.add(id);
        }
        break;
      }
    }
  }

  const standaloneRigidPairs = [];

  for (const pair of allRigidPairs) {
    let fullyContained = false;
    for (const system of rigidSubSystems) {
      if (system.includes(pair[0]) && system.includes(pair[1])) {
        fullyContained = true;
        break;
      }
    }
    if (!fullyContained) {
      standaloneRigidPairs.push(pair);
    }
  }

  return [...rigidSubSystems, ...standaloneRigidPairs];
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function isInColumnSpace(v, leftNullspace, tol = 1e-8) {
  for (let k of leftNullspace) {
    if (Math.abs(dot(k, v)) > tol) return false;
  }
  return true;
}

function getDistanceGradient(p1Index, p2Index, ptValVec) {
  const grad = new Array(ptValVec.length).fill(0);
  const x1Index = 2 * p1Index;
  const y1Index = x1Index + 1;
  const x2Index = 2 * p2Index;
  const y2Index = x2Index + 1;

  const x1 = ptValVec[x1Index];
  const y1 = ptValVec[y1Index];
  const x2 = ptValVec[x2Index];
  const y2 = ptValVec[y2Index];

  // d/d(x1): 2*(x1 - x2), d/d(y1): 2*(y1 - y2)
  // d/d(x2): 2*(x2 - x1), d/d(y2): 2*(y2 - y1)
  grad[x1Index] = 2 * (x1 - x2);
  grad[y1Index] = 2 * (y1 - y2);
  grad[x2Index] = 2 * (x2 - x1);
  grad[y2Index] = 2 * (y2 - y1);

  return grad;
}

function roundMatrix(matrix, tolerance = 1e-8) {
  matrix.forEach((row) => {
    for (let i = 0; i < row.length; i++) {
      row[i] = roundIfCloseToInteger(row[i], tolerance);
    }
  });
}

function roundIfCloseToInteger(x, tolerance) {
  const nearestInt = Math.round(x);
  return Math.abs(x - nearestInt) < tolerance ? nearestInt : x;
}
