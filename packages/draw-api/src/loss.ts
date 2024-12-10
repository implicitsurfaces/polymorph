import {
  ConstraintNode,
  ConstraintOnAngle,
  ConstraintOnDistance,
  ConstraintOnPoint,
  ConstraintOnProfileBoundary,
  findSolution,
} from "sketch";
import { point } from "./geom";
import {
  AngleLike,
  asAngle,
  asDistance,
  asDistanceOrUndefined,
  asPoint,
  asProfile,
  asRealValue,
  DistanceLike,
  PointLike,
  ProfileLike,
  RealLike,
} from "./convert";

export class LossFunction {
  constructor(private readonly terms: ConstraintNode[] = []) {}

  assertSamePoint(a: PointLike, b: PointLike, weight?: DistanceLike): void {
    this.terms.push(
      new ConstraintOnPoint(
        asPoint(a),
        asPoint(b),
        asDistanceOrUndefined(weight),
      ),
    );
  }

  assertDistance(
    distance: DistanceLike,
    target: DistanceLike,
    weight?: DistanceLike,
  ): void {
    this.terms.push(
      new ConstraintOnDistance(
        asDistance(distance),
        asDistance(target),
        asDistanceOrUndefined(weight),
      ),
    );
  }

  assertSignedDistanceToProfile(
    profile: ProfileLike,
    point: PointLike,
    signedDistance: RealLike,
    weight?: DistanceLike,
  ) {
    this.terms.push(
      new ConstraintOnProfileBoundary(
        asProfile(profile),
        asPoint(point),
        asRealValue(signedDistance),
        asDistanceOrUndefined(weight),
      ),
    );
  }

  assertDistanceBetweenPoints(
    p0: PointLike,
    p1: PointLike,
    target: DistanceLike,
    weight?: DistanceLike,
  ): void {
    const dist = point(p0).vecFrom(point(p1)).norm();

    this.terms.push(
      new ConstraintOnDistance(
        asDistance(dist),
        asDistance(target),
        asDistanceOrUndefined(weight),
      ),
    );
  }

  assertAngle(
    angle: AngleLike,
    target: AngleLike,
    weight?: DistanceLike,
  ): void {
    this.terms.push(
      new ConstraintOnAngle(
        asAngle(angle),
        asAngle(target),
        asDistanceOrUndefined(weight),
      ),
    );
  }

  assertPointsAtDistance(
    p1: PointLike,
    p2: PointLike,
    targetDistance: DistanceLike,
    weight?: DistanceLike,
  ): void {
    const dist = point(p1).vecFrom(point(p2)).norm();

    this.terms.push(
      new ConstraintOnDistance(
        asDistance(dist),
        asDistance(targetDistance),
        asDistanceOrUndefined(weight),
      ),
    );
  }

  assertPointsAtAngleWithOrigin(
    p1: PointLike,
    p2: PointLike,
    targetAngle: AngleLike,
    weight?: DistanceLike,
  ): void {
    const angle = point(p1).vecTo(point(p2)).asAngle();

    this.terms.push(
      new ConstraintOnAngle(
        asAngle(angle),
        asAngle(targetAngle),
        asDistanceOrUndefined(weight),
      ),
    );
  }

  assertPointsAtAngle(
    p1: PointLike,
    p2: PointLike,
    vertex: PointLike,
    targetAngle: AngleLike,
    weight?: DistanceLike,
  ): void {
    const vertex_ = point(vertex);

    const v1 = point(p1).vecFrom(vertex_);
    const v2 = point(p2).vecFrom(vertex_);

    const angle = v2.asAngle().subtract(v1.asAngle());

    this.terms.push(
      new ConstraintOnAngle(
        asAngle(angle),
        asAngle(targetAngle),
        asDistanceOrUndefined(weight),
      ),
    );
  }

  findMininum(
    options: {
      learningRate?: number;
      maxSteps?: number;
      tolerance?: number;
      momentum?: number;
      debug?: boolean;
    } = {},
  ) {
    return findSolution(this.terms, options);
  }
}
