import { types, flow, getSnapshot, getRoot } from "mobx-state-tree";
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

const Point = types
  .model("Point", { x: types.number, y: types.number })
  .actions((self) => ({
    moveTo([x, y]) {
      self.x = x;
      self.y = y;
    },
    remove() {
      const store = getRoot(self);
      store.removePoint(self);
    },
  }));

const AppState = types
  .model("AppState", {
    code: CodeState,
    definition: types.optional(types.number, 768),
    points: types.optional(types.array(Point), []),
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

    findClosePoint([x, y]) {
      return self.points.find((point) => {
        return Math.abs(point.x - x) < 4e-2 && Math.abs(point.y - y) < 4e-2;
      });
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
    addPoint([x, y]) {
      if (self.findClosePoint([x, y])) return;
      self.points.push({ x, y });
    },
    removePoint(point) {
      self.points.remove(point);
    },
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
          { points: self.points.map((point) => [point.x, point.y]) },
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

      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const points = self.points.map((point) => [point.x, point.y]);

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
