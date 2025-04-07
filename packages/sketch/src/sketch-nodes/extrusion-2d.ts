import {
  AngleNode,
  DistanceNode,
  ProfileNode,
  WidthModulationNode,
} from "./bases";

export class StaticWidthModulation extends WidthModulationNode {
  public readonly nodeType = "StaticWidthModulation";
  constructor(public readonly width: DistanceNode) {
    super();
  }
}

export class LinearWidthModulation extends WidthModulationNode {
  public readonly nodeType = "LinearWidthModulation";
  constructor(
    public readonly start: DistanceNode,
    public readonly end: DistanceNode,
  ) {
    super();
  }
}

export class EasedWidthModulation extends WidthModulationNode {
  public readonly nodeType = "EasedWidthModulation";
  constructor(
    public readonly start: DistanceNode,
    public readonly end: DistanceNode,
    public readonly easing: "in" | "out" | "inOut",
  ) {
    super();
  }
}

export class LinearExtrusion2DNode extends ProfileNode {
  public readonly nodeType = "LinearExtrusion2D";
  constructor(
    public readonly height: DistanceNode,
    public readonly widthModulation: WidthModulationNode,
  ) {
    super();
  }
}

export class ArcExtrusion2DNode extends ProfileNode {
  public readonly nodeType = "ArcExtrusion2D";
  constructor(
    public readonly radius: DistanceNode,
    public readonly angle: AngleNode,
    public readonly widthModulation: WidthModulationNode,
  ) {
    super();
  }
}

export type AnyExtrusion2DNode = LinearExtrusion2DNode | ArcExtrusion2DNode;
export type AnyWidthModulationNode =
  | StaticWidthModulation
  | LinearWidthModulation
  | EasedWidthModulation;
