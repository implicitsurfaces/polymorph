import { Layer, DocumentManager } from './Document.ts';
import { LayerListItem } from './LayerListItem.tsx';

import type { AutomergeUrl } from '@automerge/automerge-repo';
import { useDocument } from '@automerge/automerge-repo-react-hooks';

import { ADocument, ALayer, APoint } from './Document.ts';

interface LayersPanelProps {
  documentManager: DocumentManager;
  docUrl: AutomergeUrl;
}

export function LayersPanel({ documentManager, docUrl }: LayersPanelProps) {
  const [doc] = useDocument<ADocument>(docUrl);

  let items: JSX.Element[] = [];

  if (doc) {
    const activeLayerIndex = doc.activeLayerIndex;
    items = doc.layers.map((layer: ALayer, index: number) => (
      <LayerListItem
        key={index}
        documentManager={documentManager}
        docUrl={docUrl}
        index={index}
        isActive={index === activeLayerIndex}
        layerProperties={layer.properties}
      />
    ));
  }

  return (
    <div className="panel">
      <h2 className="panel-title">Layers</h2>
      <div className="panel-body">{items}</div>
    </div>
  );
}

export default LayersPanel;
