import {
  ElementId,
  Element,
  Point,
  EdgeElement,
  LineSegment,
  ArcFromStartTangent,
  CCurve,
  SCurve,
} from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";
import { NumberInput } from "./NumberInput.tsx";
import { SkeletonListItem } from "./SkeletonListItem.tsx";

interface PropertiesPanelProps {
  documentManager: DocumentManager;
}

export function PropertiesPanel({ documentManager }: PropertiesPanelProps) {
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const hoveredElementId = selection.hoveredElement();
  const selectedElementIds = selection.selectedElements();

  const onXChange = (point: Point) => {
    return (value: number) => {
      point.position.x = value;
      documentManager.commitChanges();
    };
  };

  const onYChange = (point: Point) => {
    return (value: number) => {
      point.position.y = value;
      documentManager.commitChanges();
    };
  };

  function getContentForPoint(point: Point) {
    return (
      <div className="panel-list-item">
        <div className="extra-zone">
          <p className="name single-line-text">Position: </p>
          <NumberInput
            idBase={`number-input::x${point.id}`}
            label="X"
            value={point.position.x}
            onChange={onXChange(point)}
          />
          <NumberInput
            idBase={`number-input::y${point.id}`}
            label="Y"
            value={point.position.y}
            onChange={onYChange(point)}
          />
        </div>
      </div>
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
    return (
      <>
        {getSkeletonItem(edge.startPoint, "Start Point:")}
        {getSkeletonItem(edge.endPoint, "End Point:")}
      </>
    );
  }

  function getContentForLineSegment(seg: LineSegment) {
    return <>{getContentForEdge(seg)}</>;
  }

  function getContentForArcFromStartTangent(arc: ArcFromStartTangent) {
    return <>{getContentForEdge(arc)}</>;
  }

  function getContentForCCurve(ccurve: CCurve) {
    return <>{getContentForEdge(ccurve)}</>;
  }

  function getContentForSCurve(scurve: SCurve) {
    return <>{getContentForEdge(scurve)}</>;
  }

  function getContentForElement(element: Element) {
    switch (element.type) {
      case "Point":
        return getContentForPoint(element);
      case "LineSegment":
        return getContentForLineSegment(element);
      case "ArcFromStartTangent":
        return getContentForArcFromStartTangent(element);
      case "CCurve":
        return getContentForCCurve(element);
      case "SCurve":
        return getContentForSCurve(element);
      case "Layer":
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
