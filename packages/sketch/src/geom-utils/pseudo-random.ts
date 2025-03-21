// a very simple pseudo-random number generator for demonstration purposes

import { asNum, Num } from "../num";

export class PseudoRandom {
  private seed: Num;

  constructor(seed: Num) {
    this.seed = seed;
  }

  next(): Num {
    this.seed = this.seed.mul(1664525).add(1013904223).mod(asNum(4294967296));
    return this.seed.div(4294967296);
  }
}

const RAND = new PseudoRandom(asNum(1234));

export function random() {
  return RAND.next();
}

export function randInit(seed: Num) {
  const rand = new PseudoRandom(seed);

  return () => rand.next();
}
