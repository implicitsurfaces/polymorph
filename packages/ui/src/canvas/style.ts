import { Selection } from "../Selection";
import { Node, SkeletonNode } from "../Document";

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

export interface PathStyleOptions {
  readonly lineWidth?: number;
  readonly stroke?: StrokeStyle | undefined;
  readonly fill?: FillStyle | undefined;
}

export class PathStyle {
  readonly lineWidth: number;
  readonly stroke?: StrokeStyle;
  readonly fill?: FillStyle;

  constructor(options: PathStyleOptions) {
    this.lineWidth = options.lineWidth ?? 1;
    this.stroke = options.stroke;
    this.fill = options.fill;
  }
}

export const backgroundColor = "#e0e0e0";

export const pointRadius = 5;
export const edgeWidth = 2;

const _shapeColor = "black";
const _constructionColor = "#ff6f34";
const _measureColor = "#c100ad";
const _selectedColor = "#4063d5";
const _hoveredColor = "#96a4d3";

/**
 * Returns an integer between 0 and 3 based on the hovered/selected state:
 *
 * ```
 *                +-------------+---------+
 *                | not hovered | hovered |
 * +--------------+-------------+---------+
 * | not selected |     0       |   1     |
 * +--------------+-------------+---------+
 * | selected     |     2       |   3     |
 * +--------------+-------------+---------+
 * ````
 */
export function getStyleIndex(isHovered: boolean, isSelected: boolean): number {
  return +isHovered + 2 * +isSelected;
}

/**
 * Returns an integer between 0 and 3 based on the hovered/selected state of
 * the given `node`.
 *
 * See `getStyleIndex()` for details.
 */
export function getNodeStyleIndex(node: Node, selection: Selection): number {
  const isHovered = selection.isHoveredNode(node);
  const isSelected = selection.isSelectedNode(node);
  return getStyleIndex(isHovered, isSelected);
}

const _shapeColors = [
  _shapeColor,
  _hoveredColor,
  _selectedColor,
  _selectedColor,
];

export function getShapeColor(styleIndex: number): string {
  return _shapeColors[styleIndex];
}

const _constructionColors = [
  _constructionColor,
  _hoveredColor,
  _selectedColor,
  _selectedColor,
];

export function getConstructionColor(styleIndex: number): string {
  return _constructionColors[styleIndex];
}

export function getSkeletonColor(
  node: SkeletonNode,
  styleIndex: number,
): string {
  switch (node.role) {
    case "shape":
      return getShapeColor(styleIndex);
    case "construction":
      return getConstructionColor(styleIndex);
  }
}

const _measureColors = [
  _measureColor,
  _hoveredColor,
  _selectedColor,
  _selectedColor,
];

export function getMeasureColor(styleIndex: number): string {
  return _measureColors[styleIndex];
}
