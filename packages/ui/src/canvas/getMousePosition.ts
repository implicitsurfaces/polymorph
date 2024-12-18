import { Vector2 } from "threejs-math";
import { Camera2 } from "./Camera2";

type IMouseEvent = MouseEvent | React.MouseEvent;

/**
 * Returns the offset, in CSS pixels, between the border box and the content box
 * of the given HTML element.
 */
// Note: we need this because while ResizeObserver provides the size of the
// content box, it does not provide its position in any coordinate systems,
// and its position can change even without its size changing (e.g., if the
// user scolls the page).
//
// Therefore, when a pointer event is triggered, the only way to reliably
// access the canvas position in viewport coordinate is to query its border
// box position via getBoundingClientRect(), and substracts the
// padding/border using getComputedStyle(). It might be possible to keep the
// latter cached, but it's probably not worth it.
//
function getBorderBoxToContentBoxOffset(element: HTMLElement): Vector2 {
  const cs = getComputedStyle(element);
  const paddingLeft = parseFloat(cs.paddingLeft);
  const paddingTop = parseFloat(cs.paddingTop);
  const borderLeft = parseFloat(cs.borderLeft);
  const borderTop = parseFloat(cs.borderTop);
  return new Vector2(paddingLeft + borderLeft, paddingTop + borderTop);
}

/**
 * Returns the position of the topleft corner of the content box of the given
 * HTML element (that is, exluding border and padding), in CSS pixels,
 * relative to the topleft corner of the browser's window
 * (= "viewport coordinates").
 */
function getContentBoxPosition(element: HTMLElement): Vector2 {
  const borderBox = element.getBoundingClientRect();
  const offset = getBorderBoxToContentBoxOffset(element);
  return new Vector2(borderBox.left, borderBox.top).add(offset);
}

/**
 * Return the position of the pointer event, in CSS pixels, relative to the
 * topleft corner of the browser's window (= "viewport coordinates").
 */
export function getMouseWindowPosition(event: IMouseEvent): Vector2 {
  return new Vector2(event.clientX, event.clientY);
}

/**
 * Return the position of the pointer event, in hardware pixels, relative to
 * the topleft corner of the content box of the given HTML element (that is,
 * exluding border and padding).
 */
export function getMouseViewPosition(
  event: IMouseEvent,
  element: HTMLElement,
): Vector2 {
  return getMouseWindowPosition(event) //
    .sub(getContentBoxPosition(element))
    .multiplyScalar(window.devicePixelRatio);
}

/**
 * Return the position of the pointer event in document coordinates, assuming the
 * document is drawn into the content box of the given HTML element using the
 * given camera.
 */
export function getMouseDocumentPosition(
  event: IMouseEvent,
  element: HTMLElement,
  camera: Camera2,
) {
  const viewToDocument = camera.viewMatrix().invert();
  return getMouseViewPosition(event, element).applyMatrix3(viewToDocument);
}
