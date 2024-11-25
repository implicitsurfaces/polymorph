import { expose } from "comlink";
import { runAsModule } from "./evaluateCode";
import { initLib } from "fidget";

const api = {
  render: async (
    code: string,
    definition: number,
  ): Promise<Uint8ClampedArray> => {
    await initLib();
    const values = await runAsModule(code);
    return await values.render(definition);
  },
};

export { api };

expose(api);
