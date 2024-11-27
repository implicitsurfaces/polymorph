import { Layer, DocumentManager } from './Document.ts';
import { LayerListItem } from './LayerListItem.tsx';

interface LayersPanelProps {
  documentManager: DocumentManager;
}

export function LayersPanel({ documentManager }: LayersPanelProps) {
  return (
    <div className="panel">
      <h2 className="panel-title">Layers</h2>
      <div className="panel-body">
        {documentManager.document().layers.map((layer: Layer, index: number) => (
          <LayerListItem
            key={index}
            documentManager={documentManager}
            index={index}
            layerProperties={layer.properties.clone()}
          />
        ))}
      </div>
    </div>
  );
}

export default LayersPanel;
