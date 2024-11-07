import { describe, expect, test } from "vitest";

import { dedupeTree, DedupeContext } from "./dedupe-tree";

class WithString {
  constructor(readonly value: string) {}
}

class Mixed {
  constructor(
    readonly a: WithString,
    readonly b: string,
  ) {}
}

class TwoNodes {
  constructor(
    readonly a: WithString | TwoNodes | Mixed,
    readonly b: WithString | TwoNodes | Mixed,
  ) {}
}

describe("dedupeTree", () => {
  test("dedupes instances", async () => {
    const a = new WithString("hello");
    const b = new WithString("hello");

    const c = new TwoNodes(a, b);
    expect(c.a).not.toBe(c.b);

    const deduped = await dedupeTree(c);

    expect(deduped.a).toBe(deduped.b);
  });

  test("does not dedupe instances with different values", async () => {
    const a = new WithString("hello");
    const b = new WithString("hello 2");

    const c = new TwoNodes(a, b);
    expect(c.a).not.toBe(c.b);

    const deduped = await dedupeTree(c);

    expect(deduped.a).not.toBe(deduped.b);
  });

  test("dedupes instances with nested objects", async () => {
    const a = new WithString("hello");
    const b = new WithString("hello");
    const c = new WithString("hello");

    const d = new TwoNodes(a, b);
    const e = new TwoNodes(a, c);

    const f = new TwoNodes(e, a);
    const g = new TwoNodes(d, a);

    const top = new TwoNodes(f, g);

    expect(top.a).not.toBe(top.b);

    const deduped = await dedupeTree(top);
    expect(deduped.a).toBe(deduped.b);
    expect(deduped.a.a.a).toBe(deduped.b.a.a);
    expect(deduped.a.a.b).toBe(deduped.b.a.a);
  });

  test("dedupes instances with mixed types", async () => {
    const a = new WithString("hello");
    const b = new WithString("hello");

    const d = new Mixed(a, "world");
    const e = new Mixed(b, "world");

    const top = new TwoNodes(d, e);

    expect(top.a).not.toBe(top.b);

    const deduped = await dedupeTree(top);
    expect(deduped.a).toBe(deduped.b);

    const c = new Mixed(a, "hello 2");
    const top2 = new TwoNodes(c, d);

    expect(top2.a).not.toBe(top2.b);
    const deduped2 = await dedupeTree(top2);
    expect(deduped2.a).not.toBe(deduped2.b);
  });

  test("dedupes instances with nested objects with diamond shapes", async () => {
    const a = new WithString("hello");
    const b1 = new Mixed(a, "world");
    const b2 = new Mixed(a, "my world");
    const t1 = new TwoNodes(b1, b2);

    const c = new WithString("hello");
    const d1 = new Mixed(c, "world");
    const d2 = new Mixed(c, "my world");
    const t2 = new TwoNodes(d1, d2);

    const top = new TwoNodes(t1, t2);

    expect(top.a).not.toBe(top.b);

    const deduped = await dedupeTree(top);
    expect(deduped.a).toBe(deduped.b);
    expect(deduped.a.a).toBe(deduped.b.a);
    expect(deduped.a.a.a).toBe(deduped.b.a.a);
  });

  test("fail to dedupe instances with nested objects with diamond shapes with a different end", async () => {
    const a = new WithString("hello");
    const b1 = new Mixed(a, "world");
    const b2 = new Mixed(a, "my world");
    const t1 = new TwoNodes(b1, b2);

    const c = new WithString("hello2");
    const d1 = new Mixed(c, "world");
    const d2 = new Mixed(c, "my world");
    const t2 = new TwoNodes(d1, d2);

    const top = new TwoNodes(t1, t2);

    expect(top.a).not.toBe(top.b);

    const deduped = await dedupeTree(top);
    expect(deduped.a).not.toBe(deduped.b);
    expect(deduped.a.a).not.toBe(deduped.b.a);
    expect(deduped.a.a.a).not.toBe(deduped.b.a.a);
  });
});

describe("DedupeContext", () => {
  class Empty {}

  test("should be equal to itself", async () => {
    const ctx = new DedupeContext();
    const a = new Empty();
    expect(await ctx.get(a)).toBe(await ctx.get(a));
  });

  test("two intances should be deduped", async () => {
    const ctx = new DedupeContext();
    const a = new Empty();
    const b = new Empty();
    expect(a).not.toBe(b);
    expect(await ctx.get(a)).toBe(await ctx.get(b));
  });

  test("instances from different classes should not be deduped", async () => {
    class Empty2 {}

    const ctx = new DedupeContext();
    const a = new Empty();
    const b = new Empty2();
    expect(a).not.toBe(b);
    expect(await ctx.get(a)).not.toBe(await ctx.get(b));
  });

  test("dedupes string values correctly", async () => {
    const ctx = new DedupeContext();
    const a = new WithString("hello");
    const b = new WithString("hello");
    expect(await ctx.get(a)).toBe(await ctx.get(b));
    const c = new WithString("world");
    expect(await ctx.get(a)).not.toBe(await ctx.get(c));
  });

  test("dedupes number values correctly", async () => {
    class WithNumber {
      constructor(public value: number) {}
    }

    const ctx = new DedupeContext();
    const a = new WithNumber(1);
    const b = new WithNumber(1);
    expect(await ctx.get(a)).toBe(await ctx.get(b));
    const c = new WithNumber(2);
    expect(await ctx.get(a)).not.toBe(await ctx.get(c));
    const d = new WithNumber(1.0);
    expect(await ctx.get(a)).toBe(await ctx.get(d));
  });

  test("dedupes boolean values correctly", async () => {
    class WithBoolean {
      constructor(public value: boolean) {}
    }

    const ctx = new DedupeContext();
    const a = new WithBoolean(true);
    const b = new WithBoolean(true);
    expect(await ctx.get(a)).toBe(await ctx.get(b));
    const c = new WithBoolean(false);
    expect(await ctx.get(a)).not.toBe(await ctx.get(c));
  });

  test("dedupes null values correctly", async () => {
    class WithNull {
      constructor(public value: null | string) {}
    }

    const ctx = new DedupeContext();
    const a = new WithNull(null);
    const b = new WithNull(null);
    expect(a).not.toBe(b);
    expect(await ctx.get(a)).toBe(await ctx.get(b));
    const c = new WithNull("ha");
    expect(await ctx.get(a)).not.toBe(await ctx.get(c));
  });

  test("dedupes undefined values correctly", async () => {
    class WithUndefined {
      constructor(public value: undefined | string | null) {}
    }

    const ctx = new DedupeContext();
    const a = new WithUndefined(undefined);
    const b = new WithUndefined(undefined);
    expect(await ctx.get(a)).toBe(await ctx.get(b));
    const c = new WithUndefined("ha");
    expect(await ctx.get(a)).not.toBe(await ctx.get(c));
    const d = new WithUndefined(null);
    expect(await ctx.get(a)).not.toBe(await ctx.get(d));
  });

  test("dedupes multiple values correctly", async () => {
    class WithMultiple {
      constructor(
        public a: string,
        public b: string,
      ) {}
    }

    const ctx = new DedupeContext();
    const a = new WithMultiple("hello", "world");
    const b = new WithMultiple("hello", "world");
    expect(await ctx.get(a)).toBe(await ctx.get(b));
    const c = new WithMultiple("world", "hello");
    expect(await ctx.get(a)).not.toBe(await ctx.get(c));
  });

  test("only dedupes nested object that have already been hashed", () => {
    const ctx = new DedupeContext();
    const a = new Mixed(new WithString("hello"), "world");

    expect(async () => await ctx.get(a)).rejects.toThrow(
      "Object has not been hashed yet",
    );
  });

  test("dedupes nested objects correctly", async () => {
    const ctx = new DedupeContext();

    const str = new WithString("hello");
    const a = new Mixed(str, "world");
    const b = new Mixed(str, "world");

    await ctx.get(str);

    expect(await ctx.get(a)).toBe(await ctx.get(b));
  });
});
