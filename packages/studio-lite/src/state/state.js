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
    processing: false,
    shapeLoaded: false,
    definition: 750,
    error: false,
    faceInfo: null,
    processingInfo: null,
    exceptionMode: "single",
  }))
  .actions((self) => ({
    updateCode(newCode) {
      self.config.code = newCode;
    },

    initCode: flow(function* () {
      const code = yield codeInit();
      self.updateCode(code);
    }),

    process: flow(function* process() {
      self.processing = true;
      try {
        console.log("processing");
        self.currentImage = yield api.render(
          self.currentValues.code,
          self.definition,
        );
        self.error = false;
      } catch (e) {
        console.error(e);
        self.error = e;
      }
      self.processing = false;
    }),
  }))
  .extend((self) => {
    let disposer = null;

    const processor = inSeries(self.process);

    const run = async () => {
      if (!self.currentValues.code) return;
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
