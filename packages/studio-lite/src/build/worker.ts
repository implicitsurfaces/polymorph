import { expose } from "comlink";
import { runAsModule } from "./evaluateCode";
import { initLib } from "fidget";
import { LossFunction } from "draw-api";

const api = {
  render: async (
    code: string,
    definition: number,
  ): Promise<{
    image: Uint8ClampedArray | null;
    valueReads: { name: string; value: number }[];
    solution: Map<string, number>;
  }> => {
    console.log("init?");
    await initLib();
    console.log("Rendering", code);
    const loss = new LossFunction();
    console.log("Loss", loss);
    let values = await runAsModule(code, loss);
    if (!Array.isArray(values)) {
      values = [values];
    }

    console.log(values);

    const solution = loss.findMininum({ debug: true });
    console.log(solution);

    const out: {
      image: Uint8ClampedArray | null;
      valueReads: { name: string; value: number }[];
      solution: Map<string, number>;
    } = {
      image: null,
      valueReads: [],
      solution: solution,
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
};

export { api };

expose(api);
