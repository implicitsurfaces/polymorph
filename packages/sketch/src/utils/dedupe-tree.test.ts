import { expect, test } from "vitest";

import { dedupeTree } from "./dedupe-tree";

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
