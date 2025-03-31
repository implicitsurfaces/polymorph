import { getProgramData } from "./getProgramData.js";
import { clustersToPlaceToCode } from "./clustersToPlaceToCode.js";

// type cluster = { [key: string]: { x: number; y: number } }[];

export function generateProg({ clusters, currentPoints, constraints }) {
  let initValues = [];
  let paramCount = 0;
  const createParam = (initValue) => {
    const str = `params[${paramCount}]`;
    initValues.push(initValue);
    paramCount++;
    return str;
  };

  const { clustersToPlace, unsolvedPts, unsolvedConstraints } = getProgramData({
    clusters,
    currentPoints,
    constraints,
  });

  console.log({ clustersToPlace, unsolvedPts, unsolvedConstraints });

  const code = clustersToPlaceToCode(clustersToPlace, createParam);

  code.push("\n// UNSOLVED PTS");
  unsolvedPts.forEach((pt) => {
    const { id, x, y, xMatch, yMatch } = pt;

    const finalX = xMatch ? `${xMatch}` : createParam(x);
    const finalY = yMatch ? `${yMatch}` : createParam(y);

    code.push(`const ${id} = pt("${id}", ${finalX}, ${finalY})`);
  });

  code.push("\n// UNSOLVED CONSTRAINTS");
  unsolvedConstraints.forEach((c) => {
    if (c.type === "angle") {
      const [p1, p2, p3, p4] = c.points;
      code.push(
        `assert(() => angle(${p1}, ${p2}, ${p3}, ${p4}, ${c.value.toFixed(
          2,
        )}))`,
      );
    }

    if (c.type === "parallel") {
      const [p1, p2, p3, p4] = c.points;
      code.push(`assert(() => angle(${p1}, ${p2}, ${p3}, ${p4}, 0))`);
    }

    if (c.type === "perpendicular") {
      const [p1, p2, p3, p4] = c.points;
      code.push(`assert(() => angle(${p1}, ${p2}, ${p3}, ${p4}, 90))`);
    }

    if (c.type === "equal") {
      const [p1, p2, p3, p4] = c.points;
      code.push(`assert(() => equal(${p1}, ${p2}, ${p3}, ${p4}))`);
    }

    if (c.type === "pointToPointDistance") {
      const [p1, p2] = c.points;
      code.push(`assert(() => dist(${p1}, ${p2}, ${c.value.toFixed(2)}))`);
    }

    if (c.type === "lineToPointDistance") {
      const [p1, p2, p3] = c.points;
      code.push(
        `assert(() => lpDist(${p1}, ${p2}, ${p3}, ${c.value.toFixed(2)}))`,
      );
    }

    if (c.type === "horizontal") {
      // TODO: need to actually check that constraint is forward subbed, may be overwritten by fixed point
      const [p0, p1] = c.points;
      if (
        unsolvedPts.some(
          (unPt) =>
            (unPt.id === p0 &&
              typeof unPt.yMatch === "string" &&
              unPt.yMatch.slice(-2) === p1) ||
            (unPt.id === p1 &&
              typeof unPt.yMatch === "string" &&
              unPt.yMatch.slice(-2) === p0),
        )
      ) {
        console.log({ unsolvedPts, type: "horz" });
        return;
      }
      code.push(`assert(() => ${p0}.y.sub(${p1}.y))`);
    }

    if (c.type === "vertical") {
      // TODO: need to actually check that constraint is forward subbed, may be overwritten by fixed point

      const [p0, p1] = c.points;
      if (
        unsolvedPts.some(
          (unPt) =>
            (unPt.id === p0 &&
              typeof unPt.xMatch === "string" &&
              unPt.xMatch.slice(-2) === p1) ||
            (unPt.id === p1 &&
              typeof unPt.xMatch === "string" &&
              unPt.xMatch.slice(-2) === p0),
        )
      ) {
        console.log({ unsolvedPts, type: "vertical" });
        return;
      }

      code.push(`assert(() => ${p0}.x.sub(${p1}.x))`);
    }

    // if (c.type === "coincident") {
    //   const [p1, p2] = c.points;
    //   code.push(`assert(() => ${p1}.x.sub(${p2}.x))`);
    //   code.push(`assert(() => ${p1}.y.sub(${p2}.y))`);
    // }
  });

  return {
    initValues,
    code,
    clustersToPlace,
  };
}
