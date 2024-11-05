import { Hasher } from "./hasher";

export const CustomHash = Symbol("CustomHash");

export interface Hashable {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
  [CustomHash]?(): string;
}

export interface CustomHashFunction {
  (): string;
}

class DedupeContext<T extends Hashable> {
  private hashes: WeakMap<T, string>;
  private cache: Map<string, T>;

  constructor() {
    this.hashes = new WeakMap();
    this.cache = new Map();
  }

  private async _createHash(obj: T): Promise<string> {
    if (this.hashes.has(obj)) return this.hashes.get(obj)!;

    if (CustomHash in obj && typeof obj[CustomHash] === "function") {
      const hash: string = obj[CustomHash]();
      this.hashes.set(obj, hash);
      return hash;
    }

    const hasher = new Hasher();
    hasher.addClass(obj);

    const keys = Object.keys(obj).sort();

    for (const key of keys) {
      const value: unknown = obj[key];
      hasher.addString(key);

      if (typeof value === "string") {
        hasher.addString(value);
      } else if (typeof value === "number") {
        hasher.addNumber(value);
      } else if (typeof value === "boolean") {
        hasher.addBoolean(value);
      } else if (value === null) {
        hasher.addNull();
      } else if (value === undefined) {
        hasher.addUndefined();
      } else if (typeof value === "object") {
        if (!this.hashes.has(value as T)) {
          throw new Error("Object has not been hashed yet");
        }
        hasher.addString(this.hashes.get(value as T)!);
      }
    }

    console.log("hashing", hasher.str);
    const hash = await hasher.done();
    this.hashes.set(obj, hash);
    return hash;
  }

  async get(tree: T): Promise<T> {
    console.log("hasing", tree);
    const hash = await this._createHash(tree);
    console.log("hashed", tree);
    if (this.cache.has(hash)) {
      return this.cache.get(hash)!;
    }
    this.cache.set(hash, tree);
    return tree;
  }
}

function isScalar(tree: Hashable): boolean {
  return typeof tree !== "object" && tree !== null;
}

function isLeaf(tree: Hashable): boolean {
  return Object.values(tree).every(isScalar);
}

async function _dedupeTree<T extends Hashable>(
  tree: T,
  context: DedupeContext<T>,
): Promise<T> {
  if (isLeaf(tree)) {
    console.log("leaf", tree);
    return await context.get(tree);
  }

  const proxyMap = new Map<string, T>();
  for (const key in tree) {
    const childNode = tree[key];
    if (isScalar(childNode)) {
      continue;
    }
    const val = await _dedupeTree(childNode as T, context);
    proxyMap.set(key, val);
  }

  const proxiedTree = new Proxy(tree, {
    get(target, prop) {
      if (proxyMap.has(prop as string)) {
        return proxyMap.get(prop as string);
      }
      return Reflect.get(target, prop);
    },
  });

  return await context.get(proxiedTree);
}

export async function dedupeTree<T extends Hashable>(tree: T): Promise<T> {
  const context = new DedupeContext<T>();
  return await _dedupeTree(tree, context);
}
