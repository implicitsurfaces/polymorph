import { loader } from "@monaco-editor/react";

import * as monaco from "monaco-editor";
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import tsWorker from "monaco-editor/esm/vs/language/typescript/ts.worker?worker";

import prettier from "prettier/standalone";
import prettierPluginBabel from "prettier/plugins/babel";
import prettierEspree from "prettier/plugins/estree";

self.MonacoEnvironment = {
  getWorker(_, label) {
    if (label === "typescript" || label === "javascript") {
      return new tsWorker();
    }
    return new editorWorker();
  },
};

const formatWithPrettier = async (value) => {
  try {
    return await prettier.format(value, {
      parser: "babel",
      plugins: [prettierPluginBabel, prettierEspree],
    });
  } catch (e) {
    console.error(e);
    return value;
  }
};

monaco.languages.registerDocumentFormattingEditProvider("javascript", {
  provideDocumentFormattingEdits: async (model) => {
    return [
      {
        range: model.getFullModelRange(),
        text: await formatWithPrettier(model.getValue()),
      },
    ];
  },
});

loader.config({ monaco });
