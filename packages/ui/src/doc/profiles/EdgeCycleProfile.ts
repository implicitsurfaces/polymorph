/*

import { EdgeCycle, EdgeCycleLike } from "./EdgeCycle";
import { Halfedge, HalfedgeId } from "./Halfedge";
import { Document } from "../Document";

export interface EdgeCycleProfileOptions extends ProfileNodeOptions {
  readonly cycle: EdgeCycleLike;
}

export class EdgeCycleProfile extends ProfileNode {
  static readonly defaultName = "Shape";

  constructor(doc: Document, id: NodeId, options: EdgeCycleProfileOptions) {
    console.log("Creating EdgeCycleProfile");
    super(doc, id, options);
    const ocycle = options.cycle;
    const cycle = ocycle instanceof EdgeCycle ? ocycle : new EdgeCycle(ocycle);
    console.log("Creating EdgeCycleProfile 2");
    this.halfedgeIds_ = cycle.halfedges.map((h) => h.id());
  }

  clone(newDoc: Document) {
    return new EdgeCycleProfile(newDoc, this.id, this);
  }

}
import {
  EdgeNode,
  EdgeNodeData,
  EdgeNodeOptions,
  ControlPoint,
} from "../EdgeNode";

*/

import { NodeId, AnyNodeData, registerNodeType } from "../Node";
import {
  ProfileNode,
  ProfileNodeData,
  ProfileNodeOptions,
} from "../ProfileNode";
import { Document } from "../Document";

import { Halfedge, HalfedgeId } from "./Halfedge";
import { EdgeCycle, EdgeCycleLike } from "./EdgeCycle";

import { asNodeIdArray } from "../dataFromAny";

function asHalfedgeIdArray(d: AnyNodeData, property: string): HalfedgeId[] {
  // Both HalfedgeId and NodeId are just string, so these are compatible typewise.
  // We just make the error message more specific.
  try {
    return asNodeIdArray(d, property);
  } catch (e) {
    if (e instanceof Error) {
      // But if
      throw Error(e.message.replace("ID", "halfedge ID"));
    } else {
      throw e;
    }
  }
}

export interface EdgeCycleProfileData extends ProfileNodeData {
  readonly halfedgeIds: HalfedgeId[];
}

export interface EdgeCycleProfileOptions extends ProfileNodeOptions {
  readonly cycle: EdgeCycleLike;
}

export class EdgeCycleProfile extends ProfileNode {
  static readonly defaultName = "Profile";

  private _data: EdgeCycleProfileData;

  get data(): EdgeCycleProfileData {
    return this._data;
  }

  setData(data: EdgeCycleProfileData) {
    this._data = data;
  }

  constructor(doc: Document, id: NodeId, data: EdgeCycleProfileData) {
    super(doc, id);
    this._data = data;
  }

  static dataFromOptions(
    doc: Document,
    options: EdgeCycleProfileOptions,
  ): EdgeCycleProfileData {
    const ocycle = options.cycle;
    const cycle = ocycle instanceof EdgeCycle ? ocycle : new EdgeCycle(ocycle);
    return {
      ...ProfileNode.dataFromOptions(doc, options),
      halfedgeIds: cycle.halfedges.map((h) => h.id()),
    };
  }

  static dataFromAny(d: AnyNodeData): EdgeCycleProfileData {
    return {
      ...ProfileNode.dataFromAny(d),
      halfedgeIds: asHalfedgeIdArray(d, "halfedgeIds"),
    };
  }

  get cycle(): EdgeCycle {
    const halfedges = this.data.halfedgeIds.map((id) => {
      const halfedge = Halfedge.fromId(id, this.doc);
      if (!halfedge) {
        throw Error(
          `The given ID (${id}) does not correspond to a known halfedge in this document.`,
        );
      }
      return halfedge;
    });
    return new EdgeCycle(halfedges);
  }
}

registerNodeType(EdgeCycleProfile);
