export function gaussJordan(matrix, colOrder = []) {
  let workingMatrix = matrix.map((row) => [...row]);

  if (colOrder.length === 0)
    colOrder = matrix[0] ? matrix[0].map((_, i) => i) : [];

  let lead = 0;
  const rowCount = workingMatrix.length;
  const colCount = workingMatrix[0].length;
  let identitySize = 0;

  for (let r = 0; r < rowCount; r++) {
    if (lead >= colCount) break;

    // Search for a pivot in the remaining columns
    let pivotFound = false;
    let pivotCol = lead;
    for (let col = lead; col < colCount; col++) {
      for (let i = r; i < rowCount; i++) {
        if (!isZero(workingMatrix[i][col])) {
          pivotFound = true;
          pivotCol = col;
          if (i !== r) {
            [workingMatrix[i], workingMatrix[r]] = [
              workingMatrix[r],
              workingMatrix[i],
            ];
          }
          break;
        }
      }
      if (pivotFound) break;
    }

    if (!pivotFound) {
      lead++;
      r--; // repeat the same row with the new lead value
      continue;
    }

    // Swap columns if pivot is not in the current lead column
    if (pivotCol !== lead) {
      for (let i = 0; i < rowCount; i++) {
        [workingMatrix[i][lead], workingMatrix[i][pivotCol]] = [
          workingMatrix[i][pivotCol],
          workingMatrix[i][lead],
        ];
      }
      [colOrder[lead], colOrder[pivotCol]] = [
        colOrder[pivotCol],
        colOrder[lead],
      ];
    }

    // Normalize the pivot row so that the pivot becomes 1
    let pivotVal = workingMatrix[r][lead];
    workingMatrix[r] = workingMatrix[r].map((val) => val / pivotVal);

    // Eliminate all other entries in the lead column
    for (let i = 0; i < rowCount; i++) {
      if (i !== r) {
        let factor = workingMatrix[i][lead];
        workingMatrix[i] = workingMatrix[i].map(
          (val, j) => val - factor * workingMatrix[r][j],
        );
      }
    }

    identitySize++;
    lead++;
  }

  roundZeros(workingMatrix);

  // Count the number of null rows (rows of all 0)
  let nullRows = workingMatrix.filter((row) =>
    row.every((val) => val === 0),
  ).length;

  return { rref: workingMatrix, colOrder, identitySize, nullRows };
}

function roundZeros(matrix) {
  matrix.forEach((row) => {
    for (let i = 0; i < row.length; i++) {
      if (isZero(row[i])) row[i] = 0;
    }
  });
}

function isZero(x) {
  return Math.abs(x) < 1e-8;
}
