import { AnyPathNode, PathEdge, PathStart } from "../../sketch-nodes";
import { Handler } from "../../sketch-nodes/types";
import {
  guardNum,
  guardPartialPath,
  guardPoint,
  guardSegmentCreator,
} from "../guards";
import { PartialPath } from "./utils/PartialPath";

export interface PathHandler<T extends AnyPathNode> extends Handler<T> {
  category: "Path";
  eval: (node: T, children: unknown[]) => PartialPath;
}

const pathStartHandler: PathHandler<PathStart> = {
  category: "Path",
  nodeType: "PathStart",
  children: (node) => {
    if (node.cornerRadius === undefined) {
      return [node.point];
    }
    return [node.point, node.cornerRadius];
  },
  eval: function evalPathStart(_, [startPoint, cornerRadius]) {
    return new PartialPath(
      guardPoint(startPoint),
      cornerRadius === undefined ? undefined : guardNum(cornerRadius),
    );
  },
};

const pathEdgeHandler: PathHandler<PathEdge> = {
  category: "Path",
  nodeType: "PathEdge",
  children: (node) => {
    if (node.cornerRadius === undefined) {
      return [node.path, node.edge, node.point];
    }
    return [node.path, node.edge, node.point, node.cornerRadius];
  },
  eval: function evalPathEdge(_, [path, edge, point, cornerRadius]) {
    return guardPartialPath(path).append(
      guardSegmentCreator(edge),
      guardPoint(point),
      cornerRadius === undefined ? undefined : guardNum(cornerRadius),
    );
  },
};

export const pathHandlers = [pathStartHandler, pathEdgeHandler] as const;
