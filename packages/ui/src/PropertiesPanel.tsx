import { Vector2 } from "threejs-math";
import { PropsWithChildren } from "react";

import {
  Node,
  Point,
  EdgeNode,
  PointToPointDistance,
  MeasureNode,
} from "./Document";
import { DocumentManager } from "./DocumentManager";
import { Vector2Input } from "./Vector2Input";
import { NumberInput } from "./NumberInput";
import { NodeListItem } from "./NodeListItem";
import { getControlPoints } from "./ControlPoint";
import { LockIcon } from "./LockIcon";

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
  const hoveredNodeId = selection.hoveredNodeId();
  const selectedNodeIds = selection.selectedNodeIds();

  function getContentForPoint(point: Point) {
    return (
      <PropertyItem name="Position">
        <Vector2Input
          getValue={() => {
            return point.position;
          }}
          setValue={(v: Vector2) => {
            point.position = v;

            documentManager.dispatchEvent("SET_POINT");
          }}
        />
      </PropertyItem>
    );
  }

  function getNodeListItem(node: Node, title: string) {
    return (
      <NodeListItem
        documentManager={documentManager}
        id={node.id}
        key={title}
        name={node.name}
        isHovered={node.id === hoveredNodeId}
        isSelected={selectedNodeIds.includes(node.id)}
        title={title}
      />
    );
  }

  function getContentForEdge(edge: EdgeNode) {
    const controlPoints = getControlPoints(edge);
    const controlPointItems = controlPoints.map((cp) => {
      return getNodeListItem(cp.point, `${cp.prettyName}:`);
    });
    return (
      <>
        {getNodeListItem(edge.startPoint, "Start Point:")}
        {getNodeListItem(edge.endPoint, "End Point:")}
        {controlPointItems}
      </>
    );
  }

  function onLockClick(measure: MeasureNode) {
    return () => {
      measure.isLocked = !measure.isLocked;

      documentManager.dispatchEvent("CHANGED_LOCK");
    };
  }

  function getContentForPointToPointDistance(d: PointToPointDistance) {
    return (
      <>
        {getNodeListItem(d.startPoint, "Start Point:")}
        {getNodeListItem(d.endPoint, "End Point:")}
        <PropertyItem name="Distance">
          <div className="input-group">
            <NumberInput label="" value={d.number.value} onChange={() => {}} />
            <LockIcon isLocked={d.isLocked} onClick={onLockClick(d)} />
          </div>
        </PropertyItem>
      </>
    );
  }

  function getContentForNode(node: Node) {
    if (node instanceof Point) {
      return getContentForPoint(node);
    } else if (node instanceof EdgeNode) {
      return getContentForEdge(node);
    } else if (node instanceof PointToPointDistance) {
      return getContentForPointToPointDistance(node);
    } else {
      return <></>;
    }
  }

  function getContent() {
    const selectedNodes = doc.getNodes(selectedNodeIds);
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
