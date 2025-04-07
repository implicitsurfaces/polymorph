import { NodeWrapper } from "./types";
import {
  debugRenderProfile,
  Difference,
  Dilate,
  exportAsFidget,
  ExtrusionNode,
  FlipNode,
  GradientAt,
  Intersection,
  MidSurfaceNode,
  Morph,
  NormalizedFieldNode,
  renderProfile,
  Rotation,
  Scale,
  Shell,
  SignedDistanceToProfile,
  SmoothDifference,
  SmoothIntersection,
  SmoothUnion,
  Translation,
  Union,
} from "sketch";
import { booleansToASCII, intArrayToImageData } from "./utils";
import {
  AngleLike,
  asAngle,
  asDistance,
  asPoint,
  asVector,
  DistanceLike,
  PointLike,
  RealLike,
  VectorLike,
} from "./convert";
import { Real, Vector, vector } from "./geom";
import { SolidEditor } from "./SolidEditor";
import { AnyProfileNode } from "sketch/dist/sketch-nodes/types";

export class ProfileEditor implements NodeWrapper<AnyProfileNode> {
  constructor(public inner: AnyProfileNode) {}

  get shape(): AnyProfileNode {
    return this.inner;
  }

  public gradientAt(p: PointLike): Vector {
    return vector(new GradientAt(this.inner, asPoint(p)));
  }

  public translate(vector: VectorLike): ProfileEditor {
    console.log("vector", vector);
    return new ProfileEditor(new Translation(this.inner, asVector(vector)));
  }

  public translateX(x: RealLike): ProfileEditor {
    return this.translate([x, 0]);
  }

  public translateY(y: RealLike): ProfileEditor {
    return this.translate([0, y]);
  }

  public rotate(theta: AngleLike): ProfileEditor {
    return new ProfileEditor(new Rotation(this.inner, asAngle(theta)));
  }

  public union(
    other: ProfileEditor,
    smoothRadius: null | DistanceLike = null,
  ): ProfileEditor {
    if (smoothRadius !== null) {
      return new ProfileEditor(
        new SmoothUnion(this.inner, other.inner, asDistance(smoothRadius)),
      );
    }
    return new ProfileEditor(new Union(this.inner, other.inner));
  }

  public fuse(
    other: ProfileEditor,
    smoothRadius: null | DistanceLike = null,
  ): ProfileEditor {
    return this.union(other, smoothRadius);
  }

  public intersect(
    other: ProfileEditor,
    smoothRadius: DistanceLike | null = null,
  ): ProfileEditor {
    if (smoothRadius !== null) {
      return new ProfileEditor(
        new SmoothIntersection(
          this.inner,
          other.inner,
          asDistance(smoothRadius),
        ),
      );
    }
    return new ProfileEditor(new Intersection(this.inner, other.inner));
  }

  public diff(
    other: ProfileEditor,
    smoothRadius: DistanceLike | null = null,
  ): ProfileEditor {
    if (smoothRadius !== null) {
      return new ProfileEditor(
        new SmoothDifference(this.inner, other.inner, asDistance(smoothRadius)),
      );
    }
    return new ProfileEditor(new Difference(this.inner, other.inner));
  }

  public cut(
    other: ProfileEditor,
    smoothRadius: DistanceLike | null = null,
  ): ProfileEditor {
    return this.diff(other, smoothRadius);
  }

  public shell(thickness: DistanceLike): ProfileEditor {
    return new ProfileEditor(new Shell(this.inner, asDistance(thickness)));
  }

  public scale(factor: DistanceLike): ProfileEditor {
    return new ProfileEditor(new Scale(this.inner, asDistance(factor)));
  }

  public morph(other: ProfileEditor, t: DistanceLike): ProfileEditor {
    return new ProfileEditor(new Morph(this.inner, other.inner, asDistance(t)));
  }

  public dilate(factor: DistanceLike): ProfileEditor {
    return new ProfileEditor(new Dilate(this.inner, asDistance(factor)));
  }

  public flip(axis: "x" | "y" = "y"): ProfileEditor {
    return new ProfileEditor(new FlipNode(this.inner, axis));
  }

  public midSurface(other: ProfileEditor): ProfileEditor {
    return new ProfileEditor(new MidSurfaceNode(this.inner, other.inner));
  }

  public normalize(): ProfileEditor {
    return new ProfileEditor(new NormalizedFieldNode(this.inner));
  }

  public distanceToPoint(p: PointLike): Real {
    return new Real(new SignedDistanceToProfile(this.inner, asPoint(p)));
  }

  public extrude(height: DistanceLike): SolidEditor {
    return new SolidEditor(new ExtrusionNode(this.inner, asDistance(height)));
  }

  async debugRender(
    size = 50,
    valuedVars?: Map<string, number>,
  ): Promise<string> {
    const render = await debugRenderProfile(this.inner, valuedVars, size);
    return booleansToASCII(intArrayToImageData(render), true);
  }

  async render(
    size = 250,
    valuedVars?: Map<string, number>,
  ): Promise<Uint8ClampedArray> {
    return renderProfile(this.inner, valuedVars, size);
  }

  async fidgetExport(): Promise<string> {
    return exportAsFidget(this.inner);
  }
}
