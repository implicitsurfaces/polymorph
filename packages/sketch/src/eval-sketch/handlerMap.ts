import { Handler } from "../sketch-nodes/types";
import { angleHandlers } from "./handlers/angle";
import { constraintHandlers } from "./handlers/constraint";
import { distanceHandlers } from "./handlers/distance";
import { edgeHandlers } from "./handlers/edge";
import { pathHandlers } from "./handlers/path";
import { planeHandlers } from "./handlers/plane";
import { pointHandlers } from "./handlers/point";
import { point3Handlers } from "./handlers/point3";
import { profileHandlers } from "./handlers/profile";
import { realValueHandlers } from "./handlers/real-value";
import { solidHandlers } from "./handlers/solid";
import { vectorHandlers } from "./handlers/vector";
import { vector3Handlers } from "./handlers/vector3";

const allHandlers = [
  ...angleHandlers,
  ...constraintHandlers,
  ...distanceHandlers,
  ...edgeHandlers,
  ...pathHandlers,
  ...planeHandlers,
  ...pointHandlers,
  ...point3Handlers,
  ...profileHandlers,
  ...realValueHandlers,
  ...solidHandlers,
  ...vectorHandlers,
  ...vector3Handlers,
] as const;

type ExtractHandlerType<H> = H extends Handler<infer N> ? N : never;
type AllHandlersTypes = ExtractHandlerType<(typeof allHandlers)[number]>;

// Create a mapped type dynamically
type NodeTypeMap = {
  [H in (typeof allHandlers)[number] as ExtractHandlerType<H>["nodeType"]]: H;
};

export type AllNodeTypes = AllHandlersTypes["nodeType"];
export type HandlerNodeType<T extends keyof NodeTypeMap> = NodeTypeMap[T];

function buildHandlers() {
  const handlersMap: NodeTypeMap = {} as NodeTypeMap;

  for (const handler of allHandlers) {
    if (handlersMap[handler.nodeType]) {
      throw new Error(
        `Duplicate handler for nodeType: ${handler.nodeType} (${handler.category})`,
      );
    }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (handlersMap as any)[handler.nodeType] = handler;
  }
  return handlersMap;
}

export const handlersMap = buildHandlers();

export function getHandler<T extends AllNodeTypes>(n: {
  nodeType: T;
}): HandlerNodeType<T> {
  return handlersMap[n.nodeType];
}
