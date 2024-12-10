import { AngleVariable, DistanceVariable, RealValueVariable } from "sketch";
import { Angle, distance, Distance, Point, point } from "./geom";

export const distanceVar = (name: string): Distance =>
  distance(new DistanceVariable(name));

export const angleVar = (name: string) => new Angle(new AngleVariable(name));

export const pointVarCartesian = (name: string): Point => {
  return point([
    new RealValueVariable(`${name}.x`),
    new RealValueVariable(`${name}.y`),
  ]);
};

export const pointVarPolar = (name: string): Point => {
  return angleVar(`${name}.angle`)
    .asVec()
    .scale(distanceVar(`${name}.radius`))
    .toPoint();
};

export const pointVar = pointVarPolar;
