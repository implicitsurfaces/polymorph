import { Vector2 } from "threejs-math";
import { PropsWithChildren } from "react";

import {
  NodeId,
  Node,
  Point,
  EdgeNode,
  LineSegment,
  ArcFromStartTangent,
  CCurve,
  SCurve,
} from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";
import { Vector2Input } from "./Vector2Input.tsx";
import { SkeletonListItem } from "./SkeletonListItem.tsx";

class ControlPointProperty {
  readonly getValue: () => Vector2;
  readonly setValue: (value: Vector2) => void;

  constructor(
    readonly name: string,
    getValueMutableRef: () => Vector2,
  ) {
    this.getValue = () => {
      return getValueMutableRef().clone();
    };
    this.setValue = (value: Vector2) => {
      getValueMutableRef().copy(value);
    };
  }
}

function getControlPointProperties(
  edge: EdgeNode,
): Array<ControlPointProperty> {
  if (edge instanceof LineSegment) {
    return [];
  }
  if (edge instanceof ArcFromStartTangent) {
    return [
      new ControlPointProperty("Control Point", () => {
        return edge.controlPoint;
      }),
    ];
  }
  if (edge instanceof CCurve) {
    return [
      new ControlPointProperty("Control Point", () => {
        return edge.controlPoint;
      }),
    ];
  }
  if (edge instanceof SCurve) {
    return [
      new ControlPointProperty("Start Control Point", () => {
        return edge.startControlPoint;
      }),
      new ControlPointProperty("End Control Point", () => {
        return edge.endControlPoint;
      }),
    ];
  }
  return [];
}

interface PropertyItemProps {
  name: string;
}

function PropertyItem({
  name,
  children,
}: PropsWithChildren<PropertyItemProps>) {
  const nameWithColon = `${name}:`;
  return (
    <div className="panel-list-item">
      <div className="extra-zone">
        <p className="name single-line-text">{nameWithColon}</p>
        {children}
      </div>
    </div>
  );
}

interface PropertiesPanelProps {
  documentManager: DocumentManager;
}

export function PropertiesPanel({ documentManager }: PropertiesPanelProps) {
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const hoveredNodeId = selection.hoveredNode();
  const selectedNodeIds = selection.selectedNodes();

  function getContentForPoint(point: Point) {
    return (
      <PropertyItem name="Position">
        <Vector2Input
          getValue={() => {
            return point.position.clone();
          }}
          setValue={(v: Vector2) => {
            point.position.copy(v);
            documentManager.commitChanges();
          }}
        />
      </PropertyItem>
    );
  }

  function getSkeletonItem(id: NodeId, title: string) {
    const point = doc.getNode(id, Point);
    if (!point) {
      return <></>;
    }
    return (
      <SkeletonListItem
        documentManager={documentManager}
        id={id}
        name={point.name}
        isHovered={id === hoveredNodeId}
        isSelected={selectedNodeIds.includes(id)}
        title={title}
      />
    );
  }

  function getContentForEdge(edge: EdgeNode) {
    const cpProps = getControlPointProperties(edge);
    const cpItems = cpProps.map((cp) => {
      return (
        <PropertyItem name={cp.name}>
          <Vector2Input
            getValue={cp.getValue}
            setValue={(v: Vector2) => {
              cp.setValue(v);
              documentManager.commitChanges();
            }}
          />
        </PropertyItem>
      );
    });
    return (
      <>
        {getSkeletonItem(edge.startPoint, "Start Point:")}
        {getSkeletonItem(edge.endPoint, "End Point:")}
        {cpItems}
      </>
    );
  }

  function getContentForNode(node: Node) {
    if (node instanceof Point) {
      return getContentForPoint(node);
    } else if (node instanceof EdgeNode) {
      return getContentForEdge(node);
    } else {
      return <></>;
    }
  }

  function getContent() {
    const doc = documentManager.document();
    const selection = documentManager.selection();
    const selectedNodes = doc.getNodes(selection.selectedNodes());
    if (selectedNodes.length === 0) {
      return (
        <div className="panel-list-item">
          <p className="name single-line-text">Type: Empty selection</p>
        </div>
      );
    } else if (selectedNodes.length === 1) {
      const node = selectedNodes[0];
      const type = Object.getPrototypeOf(node).constructor;
      return (
        <>
          <div className="panel-list-item">
            <p className="name single-line-text">Type: {type.defaultName}</p>
          </div>
          {getContentForNode(selectedNodes[0])}
        </>
      );
    } else {
      return (
        <div className="panel-list-item">
          <p className="name single-line-text">
            Type: {selectedNodes.length} selected nodes
          </p>
        </div>
      );
    }
  }

  return (
    <div className="panel">
      <h2 className="panel-title">Properties</h2>
      <div className="panel-body">{getContent()}</div>
    </div>
  );
}

export default PropertiesPanel;
