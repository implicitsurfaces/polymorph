import * as drawAPI from "draw-api";

self.drawAPI = drawAPI;

export async function buildModuleEvaluator(moduleString: string) {
  const url = URL.createObjectURL(
    new Blob([moduleString], { type: "text/javascript" }),
  );
  return await import(/* @vite-ignore */ `${url}`);
}

export async function runAsModule(code: string) {
  const module = await buildModuleEvaluator(code);

  if (module.default) return module.default();
  return module.main();
}
