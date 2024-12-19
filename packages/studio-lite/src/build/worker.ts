import { expose } from "comlink";
import { runAsModule } from "./evaluateCode";
import { initLib } from "fidget";
import { LossFunction } from "draw-api";

const api = {
  render: async (
    code: string,
    info: unknown,
    definition: number,
  ): Promise<{
    image: Uint8ClampedArray | null;
    valueReads: { name: string; value: number }[];
    solution: Map<string, number>;
    optResults: { steps: number; change: number | undefined };
  }> => {
    await initLib();
    const loss = new LossFunction();
    let values = await runAsModule(code, [loss, info]);
    if (!Array.isArray(values)) {
      values = [values];
    }

    const { solution, ...optResults } = loss.findMininum({
      maxSteps: 500,
      learningRate: 0.2,
    });

    const out: {
      image: Uint8ClampedArray | null;
      valueReads: { name: string; value: number }[];
      solution: Map<string, number>;
      optResults: { steps: number; change: number | undefined };
    } = {
      image: null,
      valueReads: [],
      solution: solution,
      optResults,
    };

    let i = 0;
    for (const value of values) {
      if (value.render) {
        out.image = await value.render(definition, solution);
      }

      if (value?.value) {
        out.valueReads.push({
          name: value.name ?? `Value ${++i}`,
          value: value.value.read(solution),
        });
      }

      if (value.read) {
        out.valueReads.push({
          name: `Value ${++i}`,
          value: value.read(solution),
        });
      }
    }

    return out;
  },

  treeRepr: async (
    code: string,
    info: unknown,
  ): Promise<{
    image: string;
    loss: string;
    valueReads: { name: string; value: string }[];
  }> => {
    const loss = new LossFunction();
    let values = await runAsModule(code, [loss, info]);
    if (!Array.isArray(values)) {
      values = [values];
    }

    const out: {
      image: string;
      loss: string;
      valueReads: { name: string; value: string }[];
    } = {
      image: "",
      valueReads: [],
      loss: loss.treeRepr(),
    };

    let i = 0;
    for (const value of values) {
      if (value.render) {
        out.image = value.treeRepr();
      }

      if (value?.value) {
        out.valueReads.push({
          name: value.name ?? `Value ${++i}`,
          value: value.value.treeRepr(),
        });
      }

      if (value.treeRepr) {
        out.valueReads.push({
          name: `Value ${++i}`,
          value: value.treeRepr(),
        });
      }
    }

    console.log("out", out);
    return out;
  },
};

export { api };

expose(api);
