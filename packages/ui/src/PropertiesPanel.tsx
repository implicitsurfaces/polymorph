import { Vector2 } from "threejs-math";
import { PropsWithChildren } from "react";

import {
  ElementId,
  Element,
  Point,
  EdgeElement,
  isEdgeElement,
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
  edge: EdgeElement,
): Array<ControlPointProperty> {
  switch (edge.type) {
    case "LineSegment": {
      return [];
    }
    case "ArcFromStartTangent": {
      return [
        new ControlPointProperty("Control Point", () => {
          return edge.controlPoint;
        }),
      ];
    }
    case "CCurve": {
      return [
        new ControlPointProperty("Control Point", () => {
          return edge.controlPoint;
        }),
      ];
    }
    case "SCurve": {
      return [
        new ControlPointProperty("Start Control Point", () => {
          return edge.startControlPoint;
        }),
        new ControlPointProperty("End Control Point", () => {
          return edge.endControlPoint;
        }),
      ];
    }
  }
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
  const hoveredElementId = selection.hoveredElement();
  const selectedElementIds = selection.selectedElements();

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

  function getSkeletonItem(id: ElementId, title: string) {
    const element = doc.getElementFromId<Point>(id);
    if (!element) {
      return <></>;
    }
    return (
      <SkeletonListItem
        documentManager={documentManager}
        id={id}
        name={element.name}
        isHovered={id === hoveredElementId}
        isSelected={selectedElementIds.includes(id)}
        title={title}
      />
    );
  }

  function getContentForEdge(edge: EdgeElement) {
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

  function getContentForElement(element: Element) {
    if (element.type === "Point") {
      return getContentForPoint(element);
    } else if (isEdgeElement(element)) {
      return getContentForEdge(element);
    } else {
      return <></>;
    }
  }

  function getContent() {
    const doc = documentManager.document();
    const selection = documentManager.selection();
    const selectedElements = doc.getElementsFromId(
      selection.selectedElements(),
    );
    if (selectedElements.length === 0) {
      return (
        <div className="panel-list-item">
          <p className="name single-line-text">Type: Empty selection</p>
        </div>
      );
    } else if (selectedElements.length === 1) {
      return (
        <>
          <div className="panel-list-item">
            <p className="name single-line-text">
              Type: {selectedElements[0].type}
            </p>
          </div>
          {getContentForElement(selectedElements[0])}
        </>
      );
    } else {
      return (
        <div className="panel-list-item">
          <p className="name single-line-text">
            Type: {selectedElements.length} selected elements
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
