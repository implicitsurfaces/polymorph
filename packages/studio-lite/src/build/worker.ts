import { expose } from "comlink";
import { runAsModule } from "./evaluateCode";

const api = {
  render: async (
    code: string,
    definition: number,
  ): Promise<Uint8ClampedArray> => {
    const values = await runAsModule(code);
    return await values.render(definition);
  },
};

export { api };

expose(api);
