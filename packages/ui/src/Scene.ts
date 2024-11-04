import { Vector2 } from 'threejs-math';

/**
 * Stores all objects in the scene.
 */
export class Scene {
  constructor(public points: Array<Vector2> = []) {}

  /**
   * Returns a new scene with the same content as this one.
   */
  clone(): Scene {
    return new this.constructor().copy(this);
  }

  /**
   * Copies the content from the source scene into this one.
   */
  copy(source: Scene): Scene {
    this.points = source.points.map(p => p.clone());
    return this;
  }

  /**
   * Adds a point to the scene.
   */
  addPoint(position: Vector2): Scene {
    this.points.push(position);
    return this;
  }
}
