export function choleskySolve(A, b) {
  const L = choleskyDecomposition(A);
  const y = forwardSubstitution(L, b);
  return backwardSubstitution(L, y);
}

function choleskyDecomposition(A) {
  const n = A.length;
  const L = new Array(n);
  for (let i = 0; i < n; i++) {
    L[i] = new Array(n).fill(0);
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L[i][k] * L[j][k];
      }
      const val = A[i][j] - sum;
      if (i === j) {
        if (val <= 0) {
          throw new Error("Matrix is not positive definite");
        }
        L[i][j] = Math.sqrt(val);
      } else {
        L[i][j] = val / L[j][j];
      }
    }
  }
  return L;
}

function forwardSubstitution(L, b) {
  const n = L.length;
  const y = new Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) {
      sum += L[i][j] * y[j];
    }
    y[i] = (b[i] - sum) / L[i][i];
  }
  return y;
}

function backwardSubstitution(L, y) {
  const n = L.length;
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) {
      // L^T[i][j] equals L[j][i]
      sum += L[j][i] * x[j];
    }
    x[i] = (y[i] - sum) / L[i][i];
  }
  return x;
}
