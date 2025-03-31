export function evalJacobian(ad) {
  const jacobian = [];

  ad.assertions.forEach((x) => {
    ad.saveTrace();
    x = x();
    ad.resetGrad();
    // ad.assert(x);
    x.grad = 1;
    const der = ad.evalGrad();
    jacobian.push(der);
    ad.restoreTrace();
  });

  return jacobian;
}
