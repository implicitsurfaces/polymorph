export function memoizeNodeEval<
  N extends object,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  T extends (node: N) => any,
>(fn: T): T {
  const idCache = new WeakMap<N, ReturnType<T>>();

  return function (node: Parameters<T>[0]): ReturnType<T> {
    if (idCache.has(node)) return idCache.get(node)!;

    const val = fn(node);
    idCache.set(node, val);

    return val;
  } as T;
}
