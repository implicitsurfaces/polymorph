import { expose } from "comlink";
import { initLib } from "fidget";

import { drawCircle } from "draw-api";

export interface SketchApi {
  render: (docJson: string, definition: number) => Promise<Uint8ClampedArray>;
}

// XXX: Uncommenting the following block comment (or just the `drawDocument` function,
// causes the following error to be thrown (and "render()" to not be logged):
//
//￼  Layer.ts:22 Uncaught ReferenceError: Cannot access 'Node' before initialization
//
// Something wrong with the imports?

/*

import { Document } from "../doc/Document";
import { ProfileNode } from "../doc/ProfileNode";
import { drawProfileNode } from "./drawProfileNode";

export function drawDocument(doc: Document) {
  // For now, we assume there is a single ProfileNode in the document,
  // and we simply build the sketch tree that corresponds to this one.
  //
  // TODO: Have some UI mechanism such that the user can choose which
  // shapes they want to render, how they want them to be rendered
  // (interior-filled with uniform color, distance field, etc.), and
  // how to compose these renders in several shapes need to
  // be drawn in the same canvas.
  //
  for (const node of doc.nodes()) {
    if (node instanceof ProfileNode) {
      return drawProfileNode(node);
    }
  }
  return undefined;
}
*/

const api: SketchApi = {
  render: async (
    _docJson: string,
    definition: number,
  ): Promise<Uint8ClampedArray> => {
    console.log("render()");
    await initLib();
    const value = drawCircle(0.5);
    const image = await value.render(definition);
    return image;

    // const doc = Document.fromJSON(docJson);
    // const value = drawDocument(doc);
    // if (value) {
    //   return await value.render(definition);
    // } else {
    //   return new Uint8ClampedArray();
    // }
  },
};

export { api };

expose(api);
