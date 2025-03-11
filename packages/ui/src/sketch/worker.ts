import { expose } from "comlink";
import { initLib } from "fidget";
import { drawCircle } from "draw-api";

const api = {
  render: async (definition: number): Promise<Uint8ClampedArray> => {
    await initLib();
    const value = drawCircle(0.5);
    const image = await value.render(definition);
    return image;
  },
};

export { api };

expose(api);
