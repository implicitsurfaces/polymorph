import { resolve } from "path";
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

export default defineConfig({
  build: {
    minify: false,
    lib: {
      // Could also be a dictionary or array of multiple entry points
      entry: resolve(__dirname, "src/main.ts"),
      name: "draw-api",
      // the proper extensions will be added
      fileName: "draw-api",
      formats: ["es"],
    },
    rollupOptions: {
      external: ["fidget"],
    },
  },
  plugins: [
    dts({
      rollupTypes: true,
      bundledPackages: ["sketch"],
    }),
  ],
});
