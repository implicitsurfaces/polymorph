import { Layer } from "../doc/Layer";
import { NodeId } from "../doc/Node";
import { DocumentManager } from "../doc/DocumentManager";
import { LayerListItem } from "./LayerListItem";

interface LayersPanelProps {
  documentManager: DocumentManager;
}

export function LayersPanel({ documentManager }: LayersPanelProps) {
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const activeLayerId = selection.activeLayerId();

  function getItem(id: NodeId, index: number) {
    const layer = doc.getNode(id, Layer);
    if (!layer) {
      return <></>;
    }
    return (
      <LayerListItem
        key={id}
        documentManager={documentManager}
        id={id}
        index={index}
        name={layer.name}
        isActive={id === activeLayerId}
      />
    );
  }

  const items = doc.layers.map((id: NodeId, index: number) =>
    getItem(id, index),
  );

  return (
    <div className="panel">
      <h2 className="panel-title">Layers</h2>
      <div className="panel-body">{items}</div>
    </div>
  );
}

export default LayersPanel;
