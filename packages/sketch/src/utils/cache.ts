export function memoizeNodeEval<N extends object, U>(
  fn: (node: N) => U,
): (node: N) => U {
  const idCache = new WeakMap<N, U>();

  return function (node: N): U {
    if (idCache.has(node)) {
      return idCache.get(node)!;
    }

    const val = fn(node);
    idCache.set(node, val);

    return val;
  };
}
