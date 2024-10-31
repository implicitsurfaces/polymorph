import { Vector2, Matrix3 } from 'threejs-math';

/**
 * Represents a 2D transformation with intuitive parameters for
 * manipulation of a 2D canvas.
 *
 * canvasSize: number of pixels in the canvas' render target
 *
 * center: 2D position, in world coordinates, which appears at the center of
 * the viewport.
 *
 * zoom: ratio between the size of an object in view coordinates (in pixels),
 * and its size in world coordinates. Example: if zoom = 2, then an object
 * which is 100-unit wide in world coordinates appears as 200 pixels on
 * screen.
 *
 * rotation: angle (in radian), between world coordinates and view
 * coordinates. Example: if angle = pi/4, then objects appear rotated 45
 * degrees anti-clockwise.
 */
export class Camera2 {
  constructor(
    public canvasSize: Vector2 = new Vector2(0, 0),
    public center: Vector2 = new Vector2(0, 0),
    public zoom: number = 1,
    public rotation: number = 0
  ) {}

  /**
   * Returns the view matrix that this camera represents, that is, the
   * transformation from world coordinates to view coordinates.
   */
  viewMatrix(): Matrix3 {
    const cx = this.center.x;
    const cy = this.center.y;
    const w = this.canvasSize.x;
    const h = this.canvasSize.y;
    const m = new Matrix3(); // identity
    m.rotate(this.rotation);
    m.scale(this.zoom, this.zoom);
    m.translate(0.5 * w - cx, 0.5 * h - cy);
    return m;
  }

  /**
   * Returns a new camera with the same properties as this one.
   */
  clone(): Camera2 {
    return new this.constructor().copy(this);
  }

  /**
   * Copies the properties from the source camera into this one.
   */
  copy(source: Camera2) {
    this.canvasSize.copy(source.canvasSize);
    this.center.copy(source.center);
    this.zoom = source.zoom;
    this.rotation = source.rotation;
    return this;
  }

  /**
   * Modifies this camera by setting its zoom to be the given value while
   * keeping the given anchor position fixed in view coordinates.
   */
  setZoomAt(anchor: Vector2, zoom: number) {
    // Compute anchor's world coords before zoom.
    const viewToWorld = this.viewMatrix().invert();
    const p = anchor.clone().applyMatrix3(viewToWorld);

    // Set new camera zoom.
    this.zoom = zoom;

    // Apply translation to keep anchor unchanged in view coords.
    p.applyMatrix3(this.viewMatrix());
    this.center.add(p).sub(anchor);
  }

  /**
   * Modifies this camera by applying the given number of steps while keeping
   * the given anchor position fixed in view coordinates.
   */
  zoomAt(anchor: Vector2, steps: number) {
    // TODO: Use predefined zoom levels table to prevent numerical drift.
    const cubicRoot2 = 1.25992104989487; // x2 zoom every 3 steps
    const newZoom = this.zoom * Math.pow(cubicRoot2, steps);
    this.setZoomAt(anchor, newZoom);
  }

  /**
   * Modifies this camera by setting its rotation to be the given value
   * (in radians) while keeping the given anchor position fixed in view
   * coordinates.
   */
  setRotationAround(anchor: Vector2, rotation: number) {
    // Compute anchor's world coords before zoom.
    const viewToWorld = this.viewMatrix().invert();
    const p = anchor.clone().applyMatrix3(viewToWorld);

    // Set new camera rotation.
    this.rotation = rotation;

    // Apply translation to keep anchor unchanged in view coords.
    p.applyMatrix3(this.viewMatrix());
    this.center.add(p).sub(anchor);
  }

  /**
   * Modifies this camera by applying a rotation of the given angle (in radians)
   * while keeping the given anchor position fixed in view coordinates.
   */
  rotateAround(anchor: Vector2, angle: number) {
    this.setRotationAround(anchor, this.rotation + angle);
  }
}
