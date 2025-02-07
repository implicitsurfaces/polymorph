import { ElementId, Layer } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";
import { SkeletonListItem } from "./SkeletonListItem.tsx";

interface SkeletonPanelProps {
  documentManager: DocumentManager;
}

export function SkeletonPanel({ documentManager }: SkeletonPanelProps) {
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const activeLayerId = selection.activeLayer();
  const hoveredElementId = selection.hoveredElement();
  const selectedElementIds = selection.selectedElements();

  function getItem(id: ElementId) {
    const element = doc.getElement(id);
    if (!element) {
      return <></>;
    }
    return (
      <SkeletonListItem
        key={id}
        documentManager={documentManager}
        id={id}
        name={element.name}
        isHovered={id === hoveredElementId}
        isSelected={selectedElementIds.includes(id)}
      />
    );
  }

  function getItems() {
    const activeLayer = doc.getElement(activeLayerId, Layer);
    if (!activeLayer) {
      return <></>;
    }
    return activeLayer.elements.map((id: ElementId) => getItem(id));
  }

  const items = getItems();

  return (
    <div className="panel">
      <h2 className="panel-title">Skeleton</h2>
      <div className="panel-body">{items}</div>
    </div>
  );
}

export default SkeletonPanel;
