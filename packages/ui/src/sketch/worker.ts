import { expose } from "comlink";
import { initLib } from "fidget";

import { Document } from "../doc/Document";
import { ProfileNode } from "../doc/ProfileNode";

import { drawProfileNode } from "./drawProfileNode";

export interface SketchApi {
  render: (doc: Document, definition: number) => Promise<Uint8ClampedArray>;
}

function drawDocument(doc: Document) {
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

const api: SketchApi = {
  render: async (
    doc: Document,
    definition: number,
  ): Promise<Uint8ClampedArray> => {
    console.log("render()", doc);
    await initLib();
    console.log("initLib", doc);
    const value = drawDocument(doc);
    if (value) {
      return await value.render(definition);
    } else {
      return new Uint8ClampedArray();
    }
  },
};

export { api };

expose(api);
