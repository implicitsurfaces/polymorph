import { Node, Layer, SkeletonNode, MeasureNode } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";
import { NodeListItem } from "./NodeListItem.tsx";

interface NodeListPanelProps {
  documentManager: DocumentManager;
  title: string;
  filter: (node: Node) => boolean;
}

export function NodeListPanel({
  documentManager,
  title,
  filter,
}: NodeListPanelProps) {
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const activeLayerId = selection.activeLayerId();
  const hoveredNodeId = selection.hoveredNodeId();
  const selectedNodeIds = selection.selectedNodeIds();

  function getItem(node: Node) {
    const id = node.id;
    return (
      <NodeListItem
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
    const filteredNodes = [];
    for (const id of activeLayer.nodes) {
      const node = doc.getNode(id);
      if (node && filter(node)) {
        filteredNodes.push(node);
      }
    }
    return filteredNodes.map((node) => getItem(node));
  }

  const items = getItems();

  return (
    <div className="panel">
      <h2 className="panel-title">{title}</h2>
      <div className="panel-body">{items}</div>
    </div>
  );
}

export default NodeListPanel;

interface SkeletonPanelProps {
  documentManager: DocumentManager;
}

export function SkeletonPanel({ documentManager }: SkeletonPanelProps) {
  const filter = (node: Node) => {
    return node instanceof SkeletonNode;
  };
  return (
    <NodeListPanel
      documentManager={documentManager}
      title="Skeleton"
      filter={filter}
    />
  );
}

interface MeasuresPanelProps {
  documentManager: DocumentManager;
}

export function MeasuresPanel({ documentManager }: MeasuresPanelProps) {
  const filter = (node: Node) => {
    return node instanceof MeasureNode;
  };
  return (
    <NodeListPanel
      documentManager={documentManager}
      title="Measures"
      filter={filter}
    />
  );
}
