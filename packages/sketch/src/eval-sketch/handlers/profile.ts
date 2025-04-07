import { ArcExtrusion2D, LinearExtrusion2D } from "../../extrusions-2d";
import {
  ClosedPath,
  OpenPath,
  Circle as CircleProfile,
  Box as BoxProfile,
  Ellipse as EllipseProfile,
  SolidSlice,
} from "../../profiles";
import {
  Normalized,
  Translation as TranslationProfile,
  Rotation as RotationProfile,
  Scaling as ScaleProfile,
  Union as UnionProfile,
  SmoothUnion as SmoothUnionProfile,
  Intersection as IntersectionProfile,
  SmoothIntersection as SmoothIntersectionProfile,
  Difference as DifferenceProfile,
  SmoothDifference as SmoothDifferenceProfile,
  Shell as ShellProfile,
  Morph as MorphProfile,
  Flip as FlipProfile,
  Dilatation,
} from "../../sdf-operations";
import { MidSurface } from "../../sdf-operations";
import {
  ArcExtrusion2DNode,
  LinearExtrusion2DNode,
  PathClose,
  PathOpenEnd,
  Circle as CircleNode,
  Box as BoxNode,
  EllipseNode,
  SolidSliceNode,
  MidSurfaceNode,
  Translation as TranslationNode,
  Rotation as RotationNode,
  Scale as ScaleNode,
  Union as UnionNode,
  SmoothUnion as SmoothUnionNode,
  Intersection as IntersectionNode,
  SmoothIntersection as SmoothIntersectionNode,
  Difference as DifferenceNode,
  SmoothDifference as SmoothDifferenceNode,
  Shell as ShellNode,
  Morph as MorphNode,
  FlipNode,
  NormalizedFieldNode,
  Dilate,
} from "../../sketch-nodes";
import { AnyProfileNode, Handler } from "../../sketch-nodes/types";
import { DistField } from "../../types";
import {
  guardAngle,
  guardNum,
  guardPartialPath,
  guardPlane,
  guardProfile,
  guardSegmentCreator,
  guardSolid,
  guardVec2,
  guardWidthModulation,
} from "../guards";

export interface ProfileHandler<T extends AnyProfileNode> extends Handler<T> {
  category: "Profile";
  eval: (profile: T, children: unknown[]) => DistField;
}

const pathCloseHandler: ProfileHandler<PathClose> = {
  category: "Profile",
  nodeType: "PathClose",
  children: (node) => [node.path, node.edge],
  eval: function evalPathClose(_, [path, edge]) {
    return new ClosedPath(
      guardPartialPath(path).close(guardSegmentCreator(edge)),
    );
  },
};

const pathOpenEndHandler: ProfileHandler<PathOpenEnd> = {
  category: "Profile",
  nodeType: "PathOpenEnd",
  children: (node) => [node.path],
  eval: function evalPathOpenEnd(_, [path]) {
    return new OpenPath(guardPartialPath(path).segments);
  },
};

const linearExtrusion2DHandler: ProfileHandler<LinearExtrusion2DNode> = {
  category: "Profile",
  nodeType: "LinearExtrusion2D",
  children: (node) => [node.height, node.widthModulation],
  eval: function evalLinearExtrusion2D(_, [height, modulation]) {
    return new LinearExtrusion2D(
      guardNum(height),
      guardWidthModulation(modulation),
    );
  },
};

const arcExtrusion2DHandler: ProfileHandler<ArcExtrusion2DNode> = {
  category: "Profile",
  nodeType: "ArcExtrusion2D",
  children: (node) => [node.radius, node.angle, node.widthModulation],
  eval: function evalArcExtrusion2D(_, [radius, angle, modulation]) {
    return new ArcExtrusion2D(
      guardNum(radius),
      guardAngle(angle),
      guardWidthModulation(modulation),
    );
  },
};

const circleHandler: ProfileHandler<CircleNode> = {
  category: "Profile",
  nodeType: "Circle",
  children: (node) => [node.radius],
  eval: function evalCircle(_, [radius]) {
    return new CircleProfile(guardNum(radius));
  },
};

const boxHandler: ProfileHandler<BoxNode> = {
  category: "Profile",
  nodeType: "Box",
  children: (node) => [node.width, node.height],
  eval: function evalBox(_, [width, height]) {
    return new BoxProfile(guardNum(width), guardNum(height));
  },
};

const ellipseHandler: ProfileHandler<EllipseNode> = {
  category: "Profile",
  nodeType: "Ellipse",
  children: (node) => [node.majorRadius, node.minorRadius],
  eval: function evalEllipse(_, [majorRadius, minorRadius]) {
    return new EllipseProfile(guardNum(majorRadius), guardNum(minorRadius));
  },
};

const solidSliceHandler: ProfileHandler<SolidSliceNode> = {
  category: "Profile",
  nodeType: "SolidSlice",
  children: (node) => [node.solid, node.plane],
  eval: function evalSolidSlice(_, [solid, plane]) {
    return new SolidSlice(guardSolid(solid), guardPlane(plane));
  },
};

const midSurfaceHandler: ProfileHandler<MidSurfaceNode> = {
  category: "Profile",
  nodeType: "MidSurface",
  children: (node) => [node.first, node.second],
  eval: function evalMidSurface(_, [first, second]) {
    return new MidSurface(guardProfile(first), guardProfile(second));
  },
};

const normalizedFieldNodeHandler: ProfileHandler<NormalizedFieldNode> = {
  category: "Profile",
  nodeType: "NormalizedField",
  children: (node) => [node.profile],
  eval: function evalNormalizedField(_, [profile]) {
    return new Normalized(guardProfile(profile));
  },
};

const translationHandler: ProfileHandler<TranslationNode> = {
  category: "Profile",
  nodeType: "Translation",
  children: (node) => [node.vector, node.profile],
  eval: function evalTranslation(_, [vector, profile]) {
    return new TranslationProfile(guardVec2(vector), guardProfile(profile));
  },
};

const rotationHandler: ProfileHandler<RotationNode> = {
  category: "Profile",
  nodeType: "Rotation",
  children: (node) => [node.angle, node.profile],
  eval: function evalRotation(_, [angle, profile]) {
    return new RotationProfile(guardAngle(angle), guardProfile(profile));
  },
};

const scaleHandler: ProfileHandler<ScaleNode> = {
  category: "Profile",
  nodeType: "Scale",
  children: (node) => [node.factor, node.profile],
  eval: function evalScale(_, [factor, profile]) {
    return new ScaleProfile(guardNum(factor), guardProfile(profile));
  },
};

const unionHandler: ProfileHandler<UnionNode> = {
  category: "Profile",
  nodeType: "Union",
  children: (node) => [node.left, node.right],
  eval: function evalUnion(_, [left, right]) {
    return new UnionProfile(guardProfile(left), guardProfile(right));
  },
};

const smoothUnionHandler: ProfileHandler<SmoothUnionNode> = {
  category: "Profile",
  nodeType: "SmoothUnion",
  children: (node) => [node.radius, node.left, node.right],
  eval: function evalSmoothUnion(_, [radius, left, right]) {
    return new SmoothUnionProfile(
      guardNum(radius),
      guardProfile(left),
      guardProfile(right),
    );
  },
};

const intersectionHandler: ProfileHandler<IntersectionNode> = {
  category: "Profile",
  nodeType: "Intersection",
  children: (node) => [node.left, node.right],
  eval: function evalIntersection(_, [left, right]) {
    return new IntersectionProfile(guardProfile(left), guardProfile(right));
  },
};

const smoothIntersectionHandler: ProfileHandler<SmoothIntersectionNode> = {
  category: "Profile",
  nodeType: "SmoothIntersection",
  children: (node) => [node.radius, node.left, node.right],
  eval: function evalSmoothIntersection(_, [radius, left, right]) {
    return new SmoothIntersectionProfile(
      guardNum(radius),
      guardProfile(left),
      guardProfile(right),
    );
  },
};

const differenceHandler: ProfileHandler<DifferenceNode> = {
  category: "Profile",
  nodeType: "Difference",
  children: (node) => [node.left, node.right],
  eval: function evalDifference(_, [left, right]) {
    return new DifferenceProfile(guardProfile(left), guardProfile(right));
  },
};

const smoothDifferenceHandler: ProfileHandler<SmoothDifferenceNode> = {
  category: "Profile",
  nodeType: "SmoothDifference",
  children: (node) => [node.radius, node.left, node.right],
  eval: function evalSmoothDifference(_, [radius, left, right]) {
    return new SmoothDifferenceProfile(
      guardNum(radius),
      guardProfile(left),
      guardProfile(right),
    );
  },
};

const shellHandler: ProfileHandler<ShellNode> = {
  category: "Profile",
  nodeType: "Shell",
  children: (node) => [node.thickness, node.profile],
  eval: function evalShell(_, [thickness, profile]) {
    return new ShellProfile(guardNum(thickness), guardProfile(profile));
  },
};

const morphHandler: ProfileHandler<MorphNode> = {
  category: "Profile",
  nodeType: "Morph",
  children: (node) => [node.t, node.start, node.end],
  eval: function evalMorph(_, [t, start, end]) {
    return new MorphProfile(
      guardNum(t),
      guardProfile(start),
      guardProfile(end),
    );
  },
};

const dilateHandler: ProfileHandler<Dilate> = {
  category: "Profile",
  nodeType: "Dilate",
  children: (node) => [node.factor, node.profile],
  eval: function evalDilate(_, [factor, profile]) {
    return new Dilatation(guardNum(factor), guardProfile(profile));
  },
};

const flipHandler: ProfileHandler<FlipNode> = {
  category: "Profile",
  nodeType: "Flip",
  children: (node) => [node.profile],
  eval: function evalFlip(node, [profile]) {
    return new FlipProfile(node.axis, guardProfile(profile));
  },
};

export const profileHandlers = [
  pathCloseHandler,
  pathOpenEndHandler,
  linearExtrusion2DHandler,
  arcExtrusion2DHandler,
  circleHandler,
  boxHandler,
  ellipseHandler,
  solidSliceHandler,
  midSurfaceHandler,
  normalizedFieldNodeHandler,
  translationHandler,
  rotationHandler,
  scaleHandler,
  unionHandler,
  smoothUnionHandler,
  intersectionHandler,
  smoothIntersectionHandler,
  differenceHandler,
  smoothDifferenceHandler,
  shellHandler,
  morphHandler,
  dilateHandler,
  flipHandler,
] as const;
