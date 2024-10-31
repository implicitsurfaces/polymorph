import { Vector2 } from 'threejs-math';

/**
 * Stores all objects in the scene.
 */
export class Scene {
  constructor(public points: Array<Vector2> = []) {}

  addPoint(position: Vector2) {
    this.points.push(position);
  }
}

// For now, we use one global scene object.
// In the future, on can imagine there could be several scenes (e.g., tabbed documents).
//
const _globalScene: Scene = new Scene();

/**
 * Returns the scene that is currently active, that is, the scene that should
 * be used in the context of a global action being triggered.
 *
 * Examples of global actions are clicking on a menu item in the global
 * menubar (Edit > Copy), or using a global keyboard shortcut (Ctrl + C).
 */
export function getActiveScene(): Scene {
  return _globalScene;
}
