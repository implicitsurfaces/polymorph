import { Point, ElementId } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";
import { SkeletonListItem } from "./SkeletonListItem.tsx";

interface SkeletonPanelProps {
  documentManager: DocumentManager;
}

export function SkeletonPanel({ documentManager }: SkeletonPanelProps) {
  const doc = documentManager.document();

  function getItem(id: ElementId) {
    const point = doc.getElementFromId<Point>(id);
    if (!point) {
      return <></>;
    }
    return (
      <SkeletonListItem
        key={id}
        documentManager={documentManager}
        id={id}
        point={point.clone()}
      />
    );
  }

  function getItems() {
    const activeLayer = documentManager.activeLayer();
    if (!activeLayer) {
      return <></>;
    }
    return activeLayer.points.map((id: ElementId) => getItem(id));
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
