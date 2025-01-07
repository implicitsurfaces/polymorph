import {
  AngleNode,
  DistanceNode,
  ProfileNode,
  WidthModulationNode,
} from "./bases";

export class StaticWidthModulation extends WidthModulationNode {
  constructor(public readonly width: DistanceNode) {
    super();
  }
}

export class LinearWidthModulation extends WidthModulationNode {
  constructor(
    public readonly start: DistanceNode,
    public readonly end: DistanceNode,
  ) {
    super();
  }
}

export class EasedWidthModulation extends WidthModulationNode {
  constructor(
    public readonly start: DistanceNode,
    public readonly end: DistanceNode,
    public readonly easing: "in" | "out" | "inOut",
  ) {
    super();
  }
}

export class LinearExtrusion2DNode extends ProfileNode {
  constructor(
    public readonly height: DistanceNode,
    public readonly widthModulation: WidthModulationNode,
  ) {
    super();
  }
}

export class ArcExtrusion2DNode extends ProfileNode {
  constructor(
    public readonly radius: DistanceNode,
    public readonly angle: AngleNode,
    public readonly widthModulation: WidthModulationNode,
  ) {
    super();
  }
}
