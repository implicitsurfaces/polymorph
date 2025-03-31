import { createMatches } from "./createMatches.js";

export function getUnsolved({
  currentPoints,
  constraints,
  seen,
  clustersToPlace,
}) {
  const unsolvedPts = [];
  const { matches, addToMatches } = createMatches(constraints, [...seen]);

  for (const id in currentPoints) {
    addToMatches(id);
  }

  Object.entries(currentPoints).forEach((pt) => {
    const id = pt[0];
    const { x, y } = pt[1];
    if (seen.has(id)) return;

    const xId = `${id}.x`;
    const yId = `${id}.y`;

    let xMatch = null;
    let yMatch = null;
    matches.forEach((set) => {
      const first = [...set][0];
      if (set.has(xId) && set.size > 1 && first !== xId) xMatch = first;
      if (set.has(yId) && set.size > 1 && first !== yId) yMatch = first;
    });

    unsolvedPts.push({ id, x, y, xMatch, yMatch });
  });

  const unsolvedConstraints = [];

  constraints.forEach((c) => {
    const solved = clustersToPlace.some((cluster) => {
      // TODO: is this ogPoint or allPts, they should be the same?
      return c.points.every((p) => cluster.allPts.has(p));
    });

    if (solved) return;

    unsolvedConstraints.push(c);
  });

  return { unsolvedPts, unsolvedConstraints };
}
