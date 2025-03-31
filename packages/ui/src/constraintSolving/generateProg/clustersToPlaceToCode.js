export function clustersToPlaceToCode(clustersToPlace, createParam) {
  const code = [];

  clustersToPlace.forEach((cluster, i) => {
    const { origin, clusterPts } = cluster;

    if (Object.keys(clusterPts).length === 0) return;

    if (i > 0) code.push("");

    code.push(`// CLUSTER ${i}`);

    Object.entries(origin).forEach((term, j) => {
      if (term[0] === "id") return;

      const [dim, [type, val]] = term;

      if (dim === "r" && cluster.fixedAngle) {
        code.push(`const c${i}_${dim} = 0`);

        return;
      }

      if (type === "param") {
        code.push(`const c${i}_${dim} = ${createParam(val)}`);
        return;
      }

      if (type === "value") {
        code.push(`const c${i}_${dim} = ${val}`);
        return;
      }
    });

    code.push("");

    for (const key in clusterPts) {
      const pt = clusterPts[key];
      const { id, dx, dy } = pt;

      code.push(
        `const ${id} = ptCluster("${id}", c${i}_x, c${i}_y, c${i}_r, ${dx}, ${dy});`,
      );
    }
  });

  return code;
}
