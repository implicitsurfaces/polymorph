/**
 * Algorithm 1 (Greedy) to compute a maximal non-over-constrained subsystem.
 *
 * PSEUDO-CODE REFERENCE (from your screenshot):
 *
 * 1: S' = ∅            // set of redundant rows
 * 2: R' = ∅            // set of non-redundant rows
 * 3: J' = ∅            // empty matrix (no rows, same columns)
 * 4: foreach row r in J in random order do
 * 5:   add r to J' at a new row
 * 6:   if rank(J') = rank(J' \ {r}) then
 * 7:       remove r from J'
 * 8:       S' = S' ∪ {r}    // row is redundant
 * 9:   else
 * 10:      R' = R' ∪ {r}    // row is non-redundant
 */

/**
 * Computes the rank of a matrix using a simple row-echelon form approach.
 * (No direct reference in the pseudo-code; this is a helper function.)
 */
function computeRank(matrix) {
  // Make a copy so we don't mutate the original
  const mat = matrix.map((row) => [...row]);
  let rowCount = mat.length;
  if (rowCount === 0) return 0;
  let colCount = mat[0].length;

  let rank = 0;
  let row = 0;

  for (let col = 0; col < colCount && row < rowCount; col++) {
    // Find pivot in this column
    let pivot = row;
    while (pivot < rowCount && isZero(mat[pivot][col])) {
      pivot++;
    }
    // No pivot in this column
    if (pivot === rowCount) {
      continue;
    }
    // Swap to put pivot row in place
    if (pivot !== row) {
      [mat[row], mat[pivot]] = [mat[pivot], mat[row]];
    }
    // Eliminate rows below
    for (let r = row + 1; r < rowCount; r++) {
      if (!isZero(mat[r][col])) {
        let factor = mat[r][col] / mat[row][col];
        for (let c = col; c < colCount; c++) {
          mat[r][c] -= factor * mat[row][c];
        }
      }
    }
    row++;
    rank++;
  }

  return rank;
}

function isZero(x) {
  return Math.abs(x) < 1e-10;
}

/**
 * Greedy algorithm to compute a maximal non-over-constrained subsystem.
 *
 * J is the Jacobian matrix (m x n). We assume each row is an array of length n.
 *
 * Returns an object:
 *   - S0: array of { index, row } for non-redundant rows
 *   - R:  array of { index, row } for redundant rows
 */
export function checkOverconstrained(J) {
  // (No direct line in pseudo-code; we need to keep track of row indices.)
  // We'll build an array of objects: { index, row }
  const indexedRows = J.map((row, i) => ({ index: i, row }));

  // [Pseudo-code line 4] "in random order":
  for (let i = indexedRows.length - 1; i > 0; i--) {
    const randIndex = Math.floor(Math.random() * (i + 1));
    [indexedRows[i], indexedRows[randIndex]] = [
      indexedRows[randIndex],
      indexedRows[i],
    ];
  }

  // [Pseudo-code lines 1 and 2] "S' = ∅" and "R' = ∅":
  let S0 = []; // will hold non-redundant rows
  let R = []; // will hold redundant rows

  // [Pseudo-code line 3] "J' = ∅":
  let Jprime = [];
  let currentRank = 0;

  // [Pseudo-code line 4 continued] for each row r in random order
  for (const { index, row } of indexedRows) {
    // [Pseudo-code line 5] add r to J' (tentatively)
    Jprime.push(row);

    // Compute rank after adding this row
    let newRank = computeRank(Jprime);

    // [Pseudo-code line 6] if rank(J') = rank(J' \ {r}) ...
    if (newRank === currentRank) {
      // [Pseudo-code line 7] remove r from J'
      Jprime.pop();
      // [Pseudo-code line 8] S' = S' ∪ {r} (i.e., it's redundant)
      R.push({ index, row });
    } else {
      // The rank increased, so the row is non-redundant
      currentRank = newRank;
      // [Pseudo-code lines 9 & 10] R' = R' ∪ {r} (non-redundant)
      S0.push({ index, row });
    }
  }

  return { S0, R };
}
