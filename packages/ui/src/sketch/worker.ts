import { expose } from "comlink";
import { initLib } from "fidget";

import { Document } from "../doc/Document";
import { ProfileNode } from "../doc/ProfileNode";
import { Point } from "../doc/Point";

import { drawProfileNode } from "./drawProfileNode";

import { Vector2 } from "threejs-math";

export interface SketchApi {
  render: (docJson: string, definition: number) => Promise<Uint8ClampedArray>;
}

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

const api: SketchApi = {
  render: async (
    docJson: string,
    definition: number,
  ): Promise<Uint8ClampedArray> => {
    // Parse JSON
    const doc = Document.fromJSON(docJson);

    // Scale down and invert Y axis
    const scaleFactor = 0.005;
    for (const node of doc.nodes()) {
      if (node instanceof Point) {
        const oldPos = node.position;
        const newPos = new Vector2(
          scaleFactor * oldPos.x,
          -scaleFactor * oldPos.y,
        );
        node.position = newPos;
      }
    }

    // Converts the document to a renderable ProfileEditor
    const value = drawDocument(doc);

    // Perform the fidget render
    await initLib();
    if (value) {
      return await value.render(definition);
    } else {
      return new Uint8ClampedArray();
    }
  },
};

export { api };

expose(api);
