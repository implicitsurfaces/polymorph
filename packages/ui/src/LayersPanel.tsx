import { Layer, ElementId } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";
import { LayerListItem } from "./LayerListItem.tsx";

interface LayersPanelProps {
  documentManager: DocumentManager;
}

export function LayersPanel({ documentManager }: LayersPanelProps) {
  const doc = documentManager.document();
  const selection = documentManager.selection();
  const activeLayerId = selection.activeLayerId();

  function getItem(id: ElementId, index: number) {
    const layer = doc.getElementFromId<Layer>(id);
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

  const items = doc.layers.map((id: ElementId, index: number) =>
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
