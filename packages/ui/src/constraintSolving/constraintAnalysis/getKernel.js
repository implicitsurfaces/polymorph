// Returns an array of basis vectors for the kernel (nullspace) of the given matrix.
export function getKernel(matrix, epsilon = 1e-8) {
  const m = matrix.length;
  if (m === 0) return [];
  const n = matrix[0].length;

  // Create a deep copy of the matrix so the input is not modified.
  const M = matrix.map((row) => row.slice());
  const pivotCols = []; // To store the pivot column index for each row in RREF

  // Compute the Reduced Row Echelon Form (RREF)
  let lead = 0;
  for (let r = 0; r < m; r++) {
    if (lead >= n) break;
    let i = r;
    // Find a pivot in the current column
    while (Math.abs(M[i][lead]) < epsilon) {
      i++;
      if (i === m) {
        i = r;
        lead++;
        if (lead === n) break;
      }
    }
    if (lead === n) break;
    // Swap row i with row r
    [M[r], M[i]] = [M[i], M[r]];

    // Scale row r to make the pivot equal to 1
    const lv = M[r][lead];
    for (let j = 0; j < n; j++) {
      M[r][j] /= lv;
    }

    // Eliminate all other entries in the pivot column
    for (let i = 0; i < m; i++) {
      if (i !== r) {
        const lv2 = M[i][lead];
        for (let j = 0; j < n; j++) {
          M[i][j] -= lv2 * M[r][j];
        }
      }
    }
    pivotCols.push(lead);
    lead++;
  }

  // Identify free columns (those not used as pivots)
  const pivotSet = new Set(pivotCols);
  const freeCols = [];
  for (let j = 0; j < n; j++) {
    if (!pivotSet.has(j)) freeCols.push(j);
  }

  const basis = [];
  // For each free variable, construct a basis vector for the kernel.
  for (const free of freeCols) {
    const vec = new Array(n).fill(0);
    vec[free] = 1; // Set the free variable to 1
    // For each pivot row, solve for the corresponding variable.
    for (let i = 0; i < pivotCols.length; i++) {
      const pivotCol = pivotCols[i];
      // In RREF, the equation is: x[pivotCol] + (coefficient)*x[free] = 0.
      // So set x[pivotCol] = -coefficient.
      vec[pivotCol] = -M[i][free];
    }
    basis.push(vec);
  }

  return basis;
}
