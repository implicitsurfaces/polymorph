// Method 1: Import directly

import { api } from "./worker";
export default api;

// Method 2: Import via comlink.
//
// Currently, this doesn't work due to the following runtime error:
// Layer.ts:22 Uncaught ReferenceError: Cannot access 'Node' before initialization

// import { SketchApi } from "./worker";
// import { wrap } from "comlink";
// import Worker from "./worker?worker";
// const api = wrap<SketchApi>(new Worker());
// export default api;
