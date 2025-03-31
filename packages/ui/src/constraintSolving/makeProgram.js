import { AutoDiff, AutoDiffNum } from "./AutoDiff.js";
import { perturbClusters } from "./constraintAnalysis/perturbClusters.js";

export function makeProgram({ currentPoints, constraints, paramValues }) {
  const perturbSolution = perturbClusters({
    currentPoints,
    constraints,
    paramValues,
  });

  const code = perturbSolution.code;
  const initValues = perturbSolution.initValues;

  console.log({ code, initValues });

  // if (STATE.makeNumerical) {
  //   let numericalCodeInit = generateNumericalProg({
  //     currentPoints,
  //     constraints,
  //   });

  //   code = numericalCodeInit.code;
  //   initValues = numericalCodeInit.initValues;

  //   console.log(numericalCodeInit);
  // }

  const fn = new Function(
    "params",
    "ptCluster",
    "add",
    "pt",
    "assert",
    // extra constraints
    "angle",
    "dist",
    "lpDist",
    "equal",
    code.join("\n"),
  );

  function constraintFn(params, { requestedValues = [] } = {}) {
    let pts = [];
    let ad = new AutoDiff();

    function addPt(id, x, y) {
      const newX = x instanceof AutoDiffNum ? x : ad.num(x);
      const newY = y instanceof AutoDiffNum ? y : ad.num(y);
      pts.push({
        id,
        x: newX,
        y: newY,
      });
      return { x: newX, y: newY };
    }

    const included = {
      ptCluster(id, ox, oy, rotate, dx, dy) {
        dx = dx instanceof AutoDiffNum ? dx : ad.num(dx);
        dy = dy instanceof AutoDiffNum ? dy : ad.num(dy);
        rotate = rotate instanceof AutoDiffNum ? rotate : ad.num(rotate);

        const rotateOriginX = dx;
        const rotateOriginY = dy;

        // Convert the rotation angle to radians
        const angleRad = rotate.mul(Math.PI / 180);

        const rotatedX = rotateOriginX
          .mul(angleRad.cos())
          .sub(rotateOriginY.mul(angleRad.sin()))
          .add(ox);
        const rotatedY = rotateOriginX
          .mul(angleRad.sin())
          .add(rotateOriginY.mul(angleRad.cos()))
          .add(oy);

        // Add the new point
        const newPt = addPt(id, rotatedX, rotatedY);
        return newPt;
      },
      add(x, y) {
        x = x instanceof AutoDiffNum ? x : ad.num(x);
        y = y instanceof AutoDiffNum ? y : ad.num(y);

        return x.add(y);
      },
      pt: addPt,
      assert(expression) {
        ad.assert(expression);
      },
      angle(p1, p2, p3, p4, value) {
        const minus = (a, b) => [a[0].sub(b[0]), a[1].sub(b[1])];
        const dot = (a, b) => a[0].mul(b[0]).add(a[1].mul(b[1]));
        const norm = (a) => {
          const x2 = a[0].pow(2);
          const y2 = a[1].pow(2);

          return x2.add(y2).sqrt();
        };

        const p1x = p1.x;
        const p1y = p1.y;
        const a = [p1x, p1y];

        const p2x = p2.x;
        const p2y = p2.y;
        const b = [p2x, p2y];

        const p3x = p3.x;
        const p3y = p3.y;
        const c = [p3x, p3y];

        const p4x = p4.x;
        const p4y = p4.y;
        const d = [p4x, p4y];

        if (value === 0) {
          // (neg(p2x) + p1x) * (p4y - p3y) + (p2y - p1y) * (p4x - p3x)
          const term1 = p2x.neg().add(p1x).mul(p4y.sub(p3y));
          const term2 = p2y.sub(p1y).mul(p4x.sub(p3x));
          return term1.add(term2);
        }
        if (value === 90) {
          // (p2x-p1x) * (p4x-p3x) + (p2y-p1y) * (p4y-p3y)

          const term1 = p2x.sub(p1x).mul(p4x.sub(p3x));
          const term2 = p2y.sub(p1y).mul(p4y.sub(p3y));
          return term1.add(term2);
        }

        const angleRads = (value / 180) * Math.PI + Math.PI / 2;
        const cosTheta = Math.cos(angleRads);
        const sinTheta = Math.sin(angleRads);

        const rotatedL2p2x = p4x
          .sub(p3x)
          .mul(cosTheta)
          .sub(p4y.sub(p3y).mul(sinTheta))
          .add(p3x);
        const rotatedL2p2y = p4x
          .sub(p3x)
          .mul(sinTheta)
          .add(p4y.sub(p3y).mul(cosTheta))
          .add(p3y);

        const dRot = [rotatedL2p2x, rotatedL2p2y];

        const numerator = dot(minus(a, b), minus(c, dRot));

        // used to prevent line from collapsing
        const denominator = norm(minus(a, b)).mul(norm(minus(c, dRot)));
        const final = numerator.div(denominator);
        return final;
      },
      dist(p1, p2, value) {
        const p1x = p1.x;
        const p1y = p1.y;
        const p2x = p2.x;
        const p2y = p2.y;

        const xTerm = p2x.sub(p1x).pow(2);
        const yTerm = p2y.sub(p1y).pow(2);
        const dist = xTerm.add(yTerm).sqrt();

        // coud just square value
        value = value instanceof AutoDiffNum ? value : ad.num(value);

        return value.sub(dist);
      },
      lpDist(p0, p1, p2, value) {
        const p1x = p0.x;
        const p1y = p0.y;
        const p2x = p1.x;
        const p2y = p1.y;
        const px = p2.x;
        const py = p2.y;

        let top = p2y
          .sub(p1y)
          .mul(px)
          .sub(p2x.sub(p1x).mul(py))
          .add(p2x.mul(p1y))
          .sub(p2y.mul(p1x))
          .pow(2)
          .sqrt();

        const bottom = p2x.sub(p1x).pow(2).add(p2y.sub(p1y).pow(2)).sqrt();

        value = value instanceof AutoDiffNum ? value : ad.num(value);

        return top.div(bottom).sub(value);
      },
      equal(p1, p2, p3, p4) {
        const p1x = p1.x;
        const p1y = p1.y;
        const p2x = p2.x;
        const p2y = p2.y;

        const xTerm1 = p2x.sub(p1x).pow(2);
        const yTerm1 = p2y.sub(p1y).pow(2);
        const dist1 = xTerm1.add(yTerm1).sqrt();

        const p3x = p3.x;
        const p3y = p3.y;
        const p4x = p4.x;
        const p4y = p4.y;

        const xTerm2 = p4x.sub(p3x).pow(2);
        const yTerm2 = p4y.sub(p3y).pow(2);
        const dist2 = xTerm2.add(yTerm2).sqrt();

        return dist1.sub(dist2);
      },
    };

    fn(
      params.map((x) => ad.param(x)),
      ...Object.values(included),
    );

    requestedValues.forEach((request) => {
      const { ptId, axis, param, value } = request;

      const pt = pts.find((p) => p.id === ptId);
      if (pt === undefined) return;
      const xOrYNum = axis === "x" ? pt.x : pt.y;

      // xOrYNum.val = value;

      ad.assert(() => {
        return xOrYNum.sub(value);
      });
    });

    return {
      vals: ad.values(),
      pts,
      code,
      initValues,
      // clusters,
      // remainingGraphs,
      paramVals: ad.params.map((x) => x.val),
      ad,
    };
  }

  return { constraintFn, initValues };
}
