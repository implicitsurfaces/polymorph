import { SketchApi } from "./worker";

// I keep this around for debugging purpose (the reloading is faster when
// working on files in the worker.)
//
// import { api } from "./worker"; /*

import { wrap } from "comlink";
import Worker from "./worker?worker";
const api = wrap<SketchApi>(new Worker());
/* */

export default api;
