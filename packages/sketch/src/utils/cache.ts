import { Hashable, hash } from "./identity";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function memoizeNodeEval<N extends Hashable, T extends (node: N) => any>(
  fn: T,
): T {
  const idCache = new WeakMap<N, ReturnType<T>>();

  return function (node: Parameters<T>[0]): ReturnType<T> {
    if (idCache.has(node)) return idCache.get(node)!;

    const val = fn(node);
    idCache.set(node, val);

    return val;
  } as T;
}

export function memoizeNodeEvalWithHash<
  N extends Hashable,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  T extends (node: N) => any,
>(fn: T): T {
  const idCache = new WeakMap<N, ReturnType<T>>();
  const hashCache = new Map<string, ReturnType<T>>();

  return function (node: Parameters<T>[0]): ReturnType<T> {
    if (idCache.has(node)) return idCache.get(node)!;

    const nodeHash = hash(node);
    if (hashCache.has(nodeHash)) return hashCache.get(nodeHash)!;

    const val = fn(node);
    idCache.set(node, val);
    hashCache.set(nodeHash, val);

    return val;
  } as T;
}
