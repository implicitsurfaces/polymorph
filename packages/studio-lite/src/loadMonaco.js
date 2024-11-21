import { loader } from "@monaco-editor/react";

import * as monaco from "monaco-editor";
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import jsonWorker from "monaco-editor/esm/vs/language/json/json.worker?worker";
import cssWorker from "monaco-editor/esm/vs/language/css/css.worker?worker";
import htmlWorker from "monaco-editor/esm/vs/language/html/html.worker?worker";
import tsWorker from "monaco-editor/esm/vs/language/typescript/ts.worker?worker";

import prettier from "prettier/standalone";
import prettierPluginBabel from "prettier/plugins/babel";
import prettierEspree from "prettier/plugins/estree";

self.MonacoEnvironment = {
  getWorker(_, label) {
    if (label === "json") {
      return new jsonWorker();
    }
    if (label === "css" || label === "scss" || label === "less") {
      return new cssWorker();
    }
    if (label === "html" || label === "handlebars" || label === "razor") {
      return new htmlWorker();
    }
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
