import { memo, useCallback, MouseEvent } from "react";
import { LayerProperties, DocumentManager } from "./Document.ts";

interface LayerListItemProps {
  documentManager: DocumentManager;
  index: number; // TODO: use some sort of unique ID instead? (e.g., if moved in hierarchy)
  isActive: boolean;
  layerProperties: LayerProperties; // we need this for memoization
}

export const LayerListItem = memo(
  function LayerListItem({
    documentManager,
    index,
    isActive,
    layerProperties,
  }: LayerListItemProps) {
    const onCreateLayer = useCallback(
      (event: MouseEvent<HTMLButtonElement>) => {
        // Click: insert after
        // Alt+Click: insert before
        const insertIndex = event.altKey ? index : index + 1;
        const prevActiveLayer = documentManager.activeLayerIndex();
        let nextActiveLayer = prevActiveLayer;
        if (insertIndex <= prevActiveLayer) {
          nextActiveLayer += 1;
        }
        documentManager.document().addLayer(insertIndex);
        documentManager.setActiveLayer(nextActiveLayer);
        documentManager.commitChanges();
      },
      [documentManager, index],
    );

    const onDeleteLayer = useCallback(() => {
      // Prevent deleting the last layer: this is important with the current
      // design (+/- buttons next to each layer), otherwise after the last
      // layer is deleted, it is impossible to create layers.
      if (documentManager.document().layers.length == 1) {
        return;
      }
      const prevActiveLayer = documentManager.activeLayerIndex();
      let nextActiveLayer = prevActiveLayer;
      if (index < prevActiveLayer) {
        nextActiveLayer -= 1;
      }
      documentManager.document().removeLayer(index);
      documentManager.setActiveLayer(nextActiveLayer);
      documentManager.commitChanges();
    }, [documentManager, index]);

    const onSelectLayer = useCallback(() => {
      documentManager.setActiveLayer(index);
    }, [documentManager, index]);

    return (
      <div
        className={
          "panel-list-item has-secret-zone" + (isActive ? " is-active" : "")
        }
      >
        <div className="secret-zone">
          <button className="single-character" onClick={onDeleteLayer}>
            -
          </button>
          <button className="single-character" onClick={onCreateLayer}>
            +
          </button>
        </div>
        <div className="highlight-zone" onClick={onSelectLayer}>
          <p className="name single-line-text">{layerProperties.name}</p>
        </div>
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
      prevProps.isActive === nextProps.isActive &&
      prevProps.layerProperties.equals(nextProps.layerProperties)
    );
  },
);

export default LayerListItem;
