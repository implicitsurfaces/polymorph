import {
  Difference,
  Dilate,
  evalProfile,
  fidgetRender,
  Intersection,
  Morph,
  ProfileNode,
  Rotation,
  Scale,
  Shell,
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
  asVector,
  DistanceLike,
  VectorLike,
} from "./convert";

export class ProfileEditor {
  constructor(public shape: ProfileNode) {}

  public translate(vector: VectorLike): ProfileEditor {
    return new ProfileEditor(new Translation(this.shape, asVector(vector)));
  }

  public translateX(x: DistanceLike): ProfileEditor {
    return this.translate([x, 0]);
  }

  public translateY(y: DistanceLike): ProfileEditor {
    return this.translate([0, y]);
  }

  public rotate(theta: AngleLike): ProfileEditor {
    return new ProfileEditor(new Rotation(this.shape, asAngle(theta)));
  }

  public union(
    other: ProfileEditor,
    smoothRadius: null | DistanceLike = null,
  ): ProfileEditor {
    if (smoothRadius !== null) {
      return new ProfileEditor(
        new SmoothUnion(this.shape, other.shape, asDistance(smoothRadius)),
      );
    }
    return new ProfileEditor(new Union(this.shape, other.shape));
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
          this.shape,
          other.shape,
          asDistance(smoothRadius),
        ),
      );
    }
    return new ProfileEditor(new Intersection(this.shape, other.shape));
  }

  public diff(
    other: ProfileEditor,
    smoothRadius: DistanceLike | null = null,
  ): ProfileEditor {
    if (smoothRadius !== null) {
      return new ProfileEditor(
        new SmoothDifference(this.shape, other.shape, asDistance(smoothRadius)),
      );
    }
    return new ProfileEditor(new Difference(this.shape, other.shape));
  }

  public cut(
    other: ProfileEditor,
    smoothRadius: DistanceLike | null = null,
  ): ProfileEditor {
    return this.diff(other, smoothRadius);
  }

  public shell(thickness: DistanceLike): ProfileEditor {
    return new ProfileEditor(new Shell(this.shape, asDistance(thickness)));
  }

  public scale(factor: DistanceLike): ProfileEditor {
    return new ProfileEditor(new Scale(this.shape, asDistance(factor)));
  }

  public morph(other: ProfileEditor, t: DistanceLike): ProfileEditor {
    return new ProfileEditor(new Morph(this.shape, other.shape, asDistance(t)));
  }

  public dilate(factor: DistanceLike): ProfileEditor {
    return new ProfileEditor(new Dilate(this.shape, asDistance(factor)));
  }

  async debugRender(size = 50): Promise<string> {
    const distField = evalProfile(this.shape);
    const render = await fidgetRender(distField, size);
    return booleansToASCII(intArrayToImageData(render), true);
  }

  async render(size = 250): Promise<Uint8ClampedArray> {
    const distField = evalProfile(this.shape);
    const render = await fidgetRender(distField, size, true);
    return new Uint8ClampedArray(render);
  }
}
