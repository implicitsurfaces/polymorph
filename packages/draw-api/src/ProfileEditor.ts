import { NodeWrapper } from "./types";
import {
  debugRenderProfile,
  Difference,
  Dilate,
  Intersection,
  Morph,
  ProfileNode,
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
  VectorLike,
} from "./convert";
import { Real } from "./geom";

export class ProfileEditor implements NodeWrapper<ProfileNode> {
  constructor(public inner: ProfileNode) {}

  get shape(): ProfileNode {
    return this.inner;
  }

  public translate(vector: VectorLike): ProfileEditor {
    return new ProfileEditor(new Translation(this.inner, asVector(vector)));
  }

  public translateX(x: DistanceLike): ProfileEditor {
    return this.translate([x, 0]);
  }

  public translateY(y: DistanceLike): ProfileEditor {
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

  public distanceToPoint(p: PointLike): Real {
    return new Real(new SignedDistanceToProfile(this.inner, asPoint(p)));
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
}
