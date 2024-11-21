import {
  AngleNode,
  Difference,
  DistanceNode,
  evalProfile,
  fidgetRender,
  Intersection,
  Morph,
  ProfileNode,
  Rotation,
  Scale,
  Shell,
  Translation,
  Union,
  VectorNode,
} from "sketch";
import { booleansToASCII, intArrayToImageData } from "./utils";
import { asAngle, asDistance, asVector } from "./convert";

export class ProfileEditor {
  constructor(public shape: ProfileNode) {}

  public translate(vector: [number, number] | VectorNode): ProfileEditor {
    return new ProfileEditor(new Translation(this.shape, asVector(vector)));
  }

  public translateX(x: number | DistanceNode): ProfileEditor {
    return this.translate([x, 0]);
  }

  public translateY(y: number | DistanceNode): ProfileEditor {
    return this.translate([0, y]);
  }

  public rotate(theta: number | AngleNode): ProfileEditor {
    return new ProfileEditor(new Rotation(this.shape, asAngle(theta)));
  }

  public union(other: ProfileEditor): ProfileEditor {
    return new ProfileEditor(new Union(this.shape, other.shape));
  }

  public fuse(other: ProfileEditor): ProfileEditor {
    return this.union(other);
  }

  public intersect(other: ProfileEditor): ProfileEditor {
    return new ProfileEditor(new Intersection(this.shape, other.shape));
  }

  public diff(other: ProfileEditor): ProfileEditor {
    return new ProfileEditor(new Difference(this.shape, other.shape));
  }

  public cut(other: ProfileEditor): ProfileEditor {
    return this.diff(other);
  }

  public shell(thickness: number | DistanceNode): ProfileEditor {
    return new ProfileEditor(new Shell(this.shape, asDistance(thickness)));
  }

  public scale(factor: number | DistanceNode): ProfileEditor {
    return new ProfileEditor(new Scale(this.shape, asDistance(factor)));
  }

  public morph(other: ProfileEditor, t: number | DistanceNode): ProfileEditor {
    return new ProfileEditor(new Morph(this.shape, other.shape, asDistance(t)));
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
