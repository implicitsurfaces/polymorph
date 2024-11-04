import { Md5 } from "ts-md5";

export const CustomHash = Symbol("CustomHash");

export interface CustomHashFunction {
  (manager: IdentityManager, parentObj: Set<Hashable>): string;
}

export class Hasher {
  private md5: Md5;
  constructor() {
    this.md5 = new Md5();
  }

  addString(str: string): Hasher {
    this.md5.appendStr(str);
    return this;
  }

  addNumber(num: number): Hasher {
    this.md5.appendStr(num.toString());
    return this;
  }

  addBoolean(bool: boolean): Hasher {
    this.md5.appendStr(bool.toString());
    return this;
  }

  addNull(): Hasher {
    this.md5.appendStr("null");
    return this;
  }

  addUndefined(): Hasher {
    this.md5.appendStr("undefined");
    return this;
  }

  addClass(obj: object): Hasher {
    this.md5.appendStr(obj.constructor.name);
    return this;
  }

  done(): string {
    return this.md5.end()?.toString() || "undefined";
  }
}

export interface Hashable {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
  [CustomHash]?(manager: IdentityManager, parentObj: Set<Hashable>): string;
}

export class IdentityManager {
  private ids: WeakMap<object, string>;
  private hashes: WeakMap<object, string>;

  constructor() {
    this.ids = new WeakMap();
    this.hashes = new WeakMap();
  }

  id(obj: object): string {
    if (!this.ids.has(obj)) {
      this.ids.set(obj, globalThis.crypto.randomUUID());
    }
    return this.ids.get(obj)!;
  }

  private _createHash(
    obj: Hashable,
    prevParentObj: Set<Hashable> = new Set(),
  ): string {
    if (this.hashes.has(obj)) return this.hashes.get(obj)!;

    const parentObj = new Set([obj, ...prevParentObj]);

    if (CustomHash in obj && typeof obj[CustomHash] === "function") {
      const hash: string = obj[CustomHash](this, parentObj);
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
        hasher.addClass(value);
        if (!parentObj.has(value)) {
          hasher.addString(this._createHash(value as Hashable, parentObj));
        } else {
          // We might want to take the hash of the path of the circular
          // dependency
          hasher.addString(":circular_dependency:");
        }
      }
    }

    const hash = hasher.done();
    this.hashes.set(obj, hash);
    return hash;
  }

  hash(obj: Hashable): string {
    if (!this.hashes.has(obj)) {
      this._createHash(obj);
    }
    return this.hashes.get(obj)!;
  }

  isEqual(obj1: Hashable, obj2: Hashable): boolean {
    return (
      obj1 === obj2 ||
      (this.hash(obj1) === this.hash(obj2) &&
        obj1.constructor === obj2.constructor)
      // We might want to do a deep comparison here
    );
  }
}

export const IDENTITY_MANAGER = new IdentityManager();

export function id(obj: object): string {
  return IDENTITY_MANAGER.id(obj);
}

export function hash(obj: Hashable): string {
  return IDENTITY_MANAGER.hash(obj);
}

export function isEqual(obj1: Hashable, obj2: Hashable): boolean {
  return IDENTITY_MANAGER.isEqual(obj1, obj2);
}

export const idHash: CustomHashFunction = function idHash(
  this: object,
  manager,
): string {
  return new Hasher().addString(manager.id(this)).done();
};
