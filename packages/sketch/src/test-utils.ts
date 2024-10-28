import { expect } from "vitest";
import { simple_eval } from "./num-tree";

export function ex(num: Num): typeof ReturnType<typeof expect> {
  return expect(simple_eval(num.n));
}
