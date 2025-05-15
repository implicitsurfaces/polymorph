import { outputSVG } from "./helpers";

import {
  circle,
  ellipse,
  hyperbola,
  parabola,
  arc,
} from "../src/conic-sections";
import { Point2D } from "../src/main";

outputSVG(
  [
    { shape: circle(0.3), color: "red" },
    { shape: ellipse(0.4, 0.8), color: "green" },
  ],
  "simple-conics.svg",
);

outputSVG(
  [
    { shape: hyperbola(0.5, 0.2, "horizontal"), color: "blue" },
    { shape: hyperbola(0.5, 0.2, "vertical"), color: "lightblue" },
  ],
  "hyperbola.svg",
);

outputSVG(
  [
    { shape: parabola(0.5, "horizontal"), color: "orange" },
    { shape: parabola(0.5, "vertical"), color: "purple" },
  ],
  "parabola.svg",
);

const p = (x: number, y: number) => new Point2D(x, y);

outputSVG(
  [
    { shape: arc(p(0.2, 0.2), p(0.4, 0.4), 0.3), color: "blue" },
    { shape: arc(p(0, 0), p(-0.1, -0.6), -1.5), color: "maroon" },
  ],
  "arc.svg",
  0.01,
  0.1,
);
