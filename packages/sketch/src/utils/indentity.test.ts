import { expect, test, describe } from "vitest";
import { IdentityManager, CustomHash, idHash } from "./identity";

describe("Identity", () => {
  class Empty {
    constructor() {}
  }

  test("should be equal", () => {
    const manager = new IdentityManager();
    const a = new Empty();
    expect(manager.id(a)).toEqual(manager.id(a));
  });

  test("should not be equal", () => {
    const manager = new IdentityManager();
    const a = new Empty();
    const b = new Empty();
    expect(manager.id(a)).not.toEqual(manager.id(b));
  });
});

describe("Hash", () => {
  class Empty {}

  test("should be equal to itself", () => {
    const manager = new IdentityManager();
    const a = new Empty();
    expect(manager.hash(a)).toEqual(manager.hash(a));
  });

  test("two intances should be equal", () => {
    const manager = new IdentityManager();
    const a = new Empty();
    const b = new Empty();
    expect(manager.hash(a)).toEqual(manager.hash(b));
  });

  test("instances from different classes should not be equal", () => {
    class Empty2 {}

    const manager = new IdentityManager();
    const a = new Empty();
    const b = new Empty2();
    expect(manager.hash(a)).not.toEqual(manager.hash(b));
  });

  test("compare string values correctly", () => {
    class WithString {
      constructor(public value: string) {}
    }

    const manager = new IdentityManager();
    const a = new WithString("hello");
    const b = new WithString("hello");
    expect(manager.hash(a)).toEqual(manager.hash(b));
    const c = new WithString("world");
    expect(manager.hash(a)).not.toEqual(manager.hash(c));
  });

  test("compare number values correctly", () => {
    class WithNumber {
      constructor(public value: number) {}
    }

    const manager = new IdentityManager();
    const a = new WithNumber(1);
    const b = new WithNumber(1);
    expect(manager.hash(a)).toEqual(manager.hash(b));
    const c = new WithNumber(2);
    expect(manager.hash(a)).not.toEqual(manager.hash(c));
    const d = new WithNumber(1.0);
    expect(manager.hash(a)).toEqual(manager.hash(d));
  });

  test("compare boolean values correctly", () => {
    class WithBoolean {
      constructor(public value: boolean) {}
    }

    const manager = new IdentityManager();
    const a = new WithBoolean(true);
    const b = new WithBoolean(true);
    expect(manager.hash(a)).toEqual(manager.hash(b));
    const c = new WithBoolean(false);
    expect(manager.hash(a)).not.toEqual(manager.hash(c));
  });

  test("compares null values correctly", () => {
    class WithNull {
      constructor(public value: null | string) {}
    }

    const manager = new IdentityManager();
    const a = new WithNull(null);
    const b = new WithNull(null);
    expect(manager.hash(a)).toEqual(manager.hash(b));
    const c = new WithNull("ha");
    expect(manager.hash(a)).not.toEqual(manager.hash(c));
  });
  test("compares undefined values correctly", () => {
    class WithUndefined {
      constructor(public value: undefined | string | null) {}
    }

    const manager = new IdentityManager();
    const a = new WithUndefined(undefined);
    const b = new WithUndefined(undefined);
    expect(manager.hash(a)).toEqual(manager.hash(b));
    const c = new WithUndefined("ha");
    expect(manager.hash(a)).not.toEqual(manager.hash(c));
    const d = new WithUndefined(null);
    expect(manager.hash(a)).not.toEqual(manager.hash(d));
  });

  test("compare multiple values correctly", () => {
    class WithMultiple {
      constructor(
        public a: string,
        public b: string,
      ) {}
    }

    const manager = new IdentityManager();
    const a = new WithMultiple("hello", "world");
    const b = new WithMultiple("hello", "world");
    expect(manager.hash(a)).toEqual(manager.hash(b));
    const c = new WithMultiple("world", "hello");
    expect(manager.hash(a)).not.toEqual(manager.hash(c));
  });

  test("compare nested objects correctly", () => {
    class WithString {
      constructor(public value: string) {}
    }
    class WithNested {
      constructor(public nested: WithString) {}
    }

    const manager = new IdentityManager();
    const a = new WithNested(new WithString("hello"));
    const b = new WithNested(new WithString("hello"));

    expect(manager.hash(a)).toEqual(manager.hash(b));
    const c = new WithNested(new WithString("world"));
    expect(manager.hash(a)).not.toEqual(manager.hash(c));
  });

  test("handles multiple references", () => {
    class WithNested {
      constructor(public a: WithNested | null) {}
    }
    class WithMultipleNested {
      constructor(
        public a: WithNested,
        public b: WithNested,
      ) {}
    }

    const manager = new IdentityManager();
    const leaf = new WithNested(null);
    const a = new WithNested(leaf);
    const b = new WithNested(leaf);
    const c = new WithMultipleNested(a, b);

    const d = new WithMultipleNested(a, a);

    expect(manager.hash(c)).toEqual(manager.hash(d));
  });

  test("handles circular references", () => {
    class WithNested {
      constructor(public a: WithNested | null) {}
    }

    const manager = new IdentityManager();
    const a = new WithNested(null);
    a.a = a;
    const b = new WithNested(null);
    b.a = b;
    expect(manager.hash(a)).toEqual(manager.hash(b));
  });

  test("can work with custom hash functions", () => {
    class WithCustomHash {
      constructor(public value: string) {}
      [CustomHash](): string {
        return "hello";
      }
    }

    const manager = new IdentityManager();
    const a = new WithCustomHash("bim");
    const b = new WithCustomHash("bam");
    expect(manager.hash(a)).toEqual(manager.hash(b));
    expect(manager.hash(a)).toEqual("hello");
  });

  test("can use the standard hash from id hash function", () => {
    class WithCustomHash {
      [CustomHash] = idHash;
    }

    const manager = new IdentityManager();
    const a = new WithCustomHash();
    const b = new WithCustomHash();

    expect(manager.hash(a)).not.toEqual(manager.hash(b));
    expect(manager.hash(a)).toEqual(manager.hash(a));
  });
});
