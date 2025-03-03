import { test, expect } from "vitest";
import { numNodeToFPCore } from "./num-to-fpcore";
import { hypot } from "../num-ops";
import { variable } from "../num";

test("hypothenuse", () => {
  const a = variable("a");
  const b = variable("b");

  const hyp = hypot(a, b);

  expect(numNodeToFPCore(hyp.n)).toEqual(
    "(FPCore (a b) (sqrt (+ (* a a) (* b b))))",
  );
});
