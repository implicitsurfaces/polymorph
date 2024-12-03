import { Num } from "./num";
import { fullDerivative } from "./num-diff";
import { naiveEval, NumNode } from "./num-tree";

export class Gradient {
  private totalDerivative: NumNode;
  constructor(num: Num) {
    this.totalDerivative = fullDerivative(num.n);
  }

  at(x: Map<string, number>) {
    const variables = Array.from(x.keys());
    const val = new Map(x);
    variables.forEach((variable) => {
      val.set(`d_${variable}`, 0);
    });

    const grad = new Map();
    variables.forEach((variable) => {
      const v = new Map(val);
      v.set(`d_${variable}`, 1);
      const derivative = naiveEval(this.totalDerivative, v);
      grad.set(variable, derivative);
    });

    return grad;
  }

  normAt(x: Map<string, number>) {
    const grad = this.at(x);
    let sum = 0;
    for (const value of grad.values()) {
      sum += value * value;
    }
    return Math.sqrt(sum);
  }
}

export function gradientDescentOpt(
  num: Num,
  initialX: Map<string, number>,
  {
    learningRate = 0.1,
    maxSteps = 100,
    tolerance = 0.0001,
    momentum = 0.1,
  } = {},
) {
  const gradient = new Gradient(num);
  let x = initialX;

  let i;
  let gradNorm;
  const velocity = new Map([...x.keys()].map((key) => [key, 0]));

  for (i = 0; i < maxSteps; i++) {
    const grad = gradient.at(x);
    const newX = new Map();

    for (const [key, value] of x.entries()) {
      const v =
        (velocity.get(key) ?? 0) * momentum - learningRate * grad.get(key);
      newX.set(key, value + v);
      velocity.set(key, v);
    }

    x = newX;
    gradNorm = gradient.normAt(x);
    if (gradNorm < tolerance) {
      break;
    }
  }

  return {
    solution: x,
    change: gradNorm,
    steps: i,
  };
}
