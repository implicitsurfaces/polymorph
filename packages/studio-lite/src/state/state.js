import { types, flow, getSnapshot } from "mobx-state-tree";
import { autorun } from "mobx";

import api from "../build/api";
import CodeState from "./code-state";

import codeInit from "./codeInit";

const inSeries = (func) => {
  let refresh;
  let currentlyRunning = false;

  return async function () {
    if (currentlyRunning) {
      refresh = true;
      return;
    }
    currentlyRunning = true;

    while (true) {
      refresh = false;
      await func();

      if (!refresh) break;
    }

    currentlyRunning = false;
  };
};

const AppState = types
  .model("AppState", {
    code: CodeState,
    definition: types.optional(types.number, 768),
    config: types.optional(
      types.model({
        code: types.optional(types.string, ""),
      }),
      {},
    ),
  })
  .views((self) => ({
    get currentValues() {
      return getSnapshot(self.config);
    },
    get hasError() {
      return !!self.error.error;
    },

    get codeInitialized() {
      return !!self.config.code;
    },
  }))
  .volatile(() => ({
    currentImage: null,
    valueReads: [],
    currentDefinition: null,
    processing: false,
    shapeLoaded: false,
    error: false,
    faceInfo: null,
    processingInfo: null,
    exceptionMode: "single",
  }))
  .actions((self) => ({
    updateCode(newCode) {
      self.config.code = newCode;
    },

    changeDefinition(newDefinition) {
      self.definition = newDefinition;
    },

    initCode: flow(function* () {
      const code = yield codeInit();
      self.updateCode(code);
    }),

    process: flow(function* process() {
      self.processing = true;
      console.log("Processing...");
      try {
        const results = yield api.render(
          self.currentValues.code,
          self.definition,
        );
        console.log("Results", results);
        self.currentImage = results.image;
        self.valueReads = results.valueReads;
        self.currentDefinition = self.definition;
        self.error = false;
      } catch (e) {
        console.error(e);
        console.log("Error", e);
        self.error = e;
      }

      console.log("Processed...");
      self.processing = false;
    }),
  }))
  .extend((self) => {
    let disposer = null;

    const processor = inSeries(self.process);

    const run = async () => {
      if (!self.currentValues.code) return;
      if (!self.definition) return;

      await processor();
    };

    return {
      actions: {
        afterCreate() {
          disposer = autorun(run);
        },

        afterDestroy() {
          if (disposer) disposer();
        },
      },
    };
  });

export default AppState;
