import { getUnsolved } from "./getUnsolved.js";

export function getProgramData({ clusters, currentPoints, constraints }) {
  const clustersToPlace = [];
  const allKnownValues = new Set();

  clusters.forEach((cluster, i) => {
    let fixed = constraints.find(
      (c) => c.type === "fixed" && c.points.every((p) => p in cluster),
    );

    // TODO: note if there are two fixed points in a cluster

    const clusterToPlace = {
      origin: null,
      fixedOrigin: fixed ?? null,
      clusterPts: {},
      allPts: new Set(),
      ogPoints: cluster,
      fixedAngle: constraints.some(
        (c) =>
          (c.type === "horizontal" || c.type === "vertical") &&
          c.points.every((p) => p in cluster),
      ),
    };

    clustersToPlace.push(clusterToPlace);
  });

  clustersToPlace.sort((a, b) =>
    a.fixedOrigin === null ? 1 : b.fixedOrigin === null ? -1 : 0,
  );

  clustersToPlace.forEach((cluster) => {
    let { overlap, origin } = getOverlap({
      pts: Object.entries(cluster.ogPoints),
      allKnownValues,
    });

    const { fixedOrigin } = cluster;

    cluster.origin = fixedOrigin
      ? {
          id: null,
          x: ["value", fixedOrigin.x],
          y: ["value", fixedOrigin.y],
          r: ["param", 0],
        }
      : origin;

    Object.entries(cluster.ogPoints).forEach((geo, j) => {
      const [id, val] = geo;

      let { x, y } = val;
      if (allKnownValues.has(id)) return;
      if (id === overlap?.id && !fixedOrigin) return;

      if (overlap && !fixedOrigin) {
        x = x - overlap.x;
        y = y - overlap.y;
      }

      if (fixedOrigin) {
        x = x - fixedOrigin.x;
        y = y - fixedOrigin.y;
      }

      cluster.clusterPts[id] = {
        id,
        dx: x.toFixed(2),
        dy: y.toFixed(2),
      };
    });

    const clusterPtsKeys = Object.keys(cluster.clusterPts);
    if (clusterPtsKeys.length === 0) return;

    clusterPtsKeys.forEach((id) => allKnownValues.add(id));

    cluster.allPts = new Set(
      [...clusterPtsKeys, cluster?.origin?.id].filter(
        (x) => typeof x === "string",
      ),
    );
  });

  console.log({ clustersToPlace });

  const { unsolvedPts, unsolvedConstraints } = getUnsolved({
    currentPoints,
    constraints,
    seen: allKnownValues,
    clustersToPlace,
  });

  return {
    clustersToPlace,
    unsolvedPts,
    unsolvedConstraints,
  };
}

function getOverlap({ pts, allKnownValues }) {
  let overlap = null;
  pts.forEach((geo) => {
    const [id, val] = geo;

    if (allKnownValues.has(id)) {
      overlap = {
        id,
        x: val.x,
        y: val.y,
      };
    }
  });

  let origin = null;
  if (overlap && pts.length > 1) {
    origin = {
      id: overlap.id,
      x: ["value", `${overlap.id}.x`],
      y: ["value", `${overlap.id}.y`],
      r: ["param", 0],
    };
  } else {
    origin = {
      id: null,
      x: ["param", 0],
      y: ["param", 0],
      r: ["param", 0],
    };
  }

  return {
    overlap,
    origin,
  };
}
