import { memo } from 'react';
import { LayerProperties, DocumentManager } from './Document.ts';

interface LayerListItemProps {
  documentManager: DocumentManager;
  index: number; // TODO: use some sort of unique ID instead? (e.g., if moved in hierarchy)
  layerProperties: LayerProperties; // we need this for memoization
}

export const LayerListItem = memo(
  function LayerListItem({ layerProperties }: LayerListItemProps) {
    return (
      <div className="panel-list-item">
        <p className="name">{layerProperties.name}</p>
      </div>
    );
  },
  (prevProps, nextProps) => {
    // We need re-rendering only if the layer's properties (name, color, etc.)
    // change, but not if the layer's inner objects change.
    //
    return (
      prevProps.documentManager === nextProps.documentManager &&
      prevProps.index === nextProps.index &&
      prevProps.layerProperties.equals(nextProps.layerProperties)
    );
  }
);

export default LayerListItem;
