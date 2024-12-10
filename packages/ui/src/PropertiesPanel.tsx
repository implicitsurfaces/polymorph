import { Element, Point } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";
import { NumberInput } from "./NumberInput.tsx";

interface PropertiesPanelProps {
  documentManager: DocumentManager;
}

export function PropertiesPanel({ documentManager }: PropertiesPanelProps) {
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

  function getContentForElement(element: Element) {
    if (element.type === "Point") {
      return getContentForPoint(element as Point);
    } else {
      return <></>;
    }
  }

  function getContent() {
    const selectedElements = documentManager.selectedElements();
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
