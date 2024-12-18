// A StrokeStyle or FillStyle is:
// - A string parsed as CSS <color> value.
// - A CanvasGradient object (a linear or radial gradient).
// - A CanvasPattern object (a repeating image).
//
// https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/fillStyle
// https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/strokeStyle
//
export type FillStyle = string | CanvasGradient | CanvasPattern;
export type StrokeStyle = string | CanvasGradient | CanvasPattern;

export const pointRadius = 5;
export const controlPointRadius = 4;
export const edgeWidth = 2;

const _elementColor = "black";
const _controlColor = "#ff6f34";
const _selectedColor = "#4063d5";
const _hoveredColor = "#96a4d3";

export function getElementColor(
  isHovered: boolean = false,
  isSelected: boolean = false,
): string {
  if (isSelected) {
    return _selectedColor;
  } else if (isHovered) {
    return _hoveredColor;
  } else {
    return _elementColor;
  }
}

export function getControlColor(
  isHovered: boolean = false,
  isSelected: boolean = false,
): string {
  if (isSelected) {
    return _selectedColor;
  } else if (isHovered) {
    return _hoveredColor;
  } else {
    return _controlColor;
  }
}
