import { Layer, ElementId } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";
import { LayerListItem } from "./LayerListItem.tsx";

interface LayersPanelProps {
  documentManager: DocumentManager;
}

export function LayersPanel({ documentManager }: LayersPanelProps) {
  const activeLayerId = documentManager.activeLayerId();
  const doc = documentManager.document();

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
        isActive={id === activeLayerId}
        layerProperties={layer.properties.clone()}
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
