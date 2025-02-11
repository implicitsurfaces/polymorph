import { NodeId, Layer, SkeletonNode } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";
import { SkeletonListItem } from "./SkeletonListItem.tsx";

interface SkeletonPanelProps {
  documentManager: DocumentManager;
}

export function SkeletonPanel({ documentManager }: SkeletonPanelProps) {
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const activeLayerId = selection.activeLayer();
  const hoveredNodeId = selection.hoveredNode();
  const selectedNodeIds = selection.selectedNodes();

  function getItem(id: NodeId) {
    const node = doc.getNode(id);
    if (!(node instanceof SkeletonNode)) {
      return <></>;
    }
    return (
      <SkeletonListItem
        key={id}
        documentManager={documentManager}
        id={id}
        name={node.name}
        isHovered={id === hoveredNodeId}
        isSelected={selectedNodeIds.includes(id)}
      />
    );
  }

  function getItems() {
    const activeLayer = doc.getNode(activeLayerId, Layer);
    if (!activeLayer) {
      return <></>;
    }
    return activeLayer.nodes.map((id: NodeId) => getItem(id));
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
