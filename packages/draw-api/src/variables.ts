import { AngleVariable, DistanceVariable, RealValueVariable } from "sketch";
import { Angle, distance, Distance, point } from "./geom";

export const distanceVar = (name: string): Distance =>
  distance(new DistanceVariable(name));

export const angleVar = (name: string) => new Angle(new AngleVariable(name));

export const pointVar = (name: string) => {
  return point([
    new RealValueVariable(`${name}.x`),
    new RealValueVariable(`${name}.y`),
  ]);
};
