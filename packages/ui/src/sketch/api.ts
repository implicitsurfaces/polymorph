// Method 1: Use API synchronously in the main thread

// import { api } from "./worker";
// export default api;

// Method 2: Use API asynchronously in a separate WebWorker

import { SketchApi } from "./worker";
import { wrap } from "comlink";
import Worker from "./worker?worker";
const api = wrap<SketchApi>(new Worker());
export default api;
