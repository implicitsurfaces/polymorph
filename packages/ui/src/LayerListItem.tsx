import { memo, useCallback, MouseEvent } from "react";
import { LayerProperties, ElementId } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";

interface LayerListItemProps {
  documentManager: DocumentManager;
  id: ElementId;
  index: number;
  isActive: boolean;
  layerProperties: LayerProperties; // we need this for memoization
}

export const LayerListItem = memo(
  function LayerListItem({
    documentManager,
    id,
    index,
    isActive,
    layerProperties,
  }: LayerListItemProps) {
    const onCreateLayer = useCallback(
      (event: MouseEvent<HTMLButtonElement>) => {
        // Click: insert after
        // Alt+Click: insert before
        const insertIndex = event.altKey ? index : index + 1;
        documentManager.document().createLayerAtIndex(insertIndex);
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
      documentManager.document().deleteLayerAtIndex(index);
      documentManager.commitChanges();
    }, [documentManager, index]);

    const onSelectLayer = useCallback(() => {
      documentManager.setActiveLayer(id);
    }, [documentManager, id]);

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
      prevProps.id === nextProps.id &&
      prevProps.index === nextProps.index &&
      prevProps.isActive === nextProps.isActive &&
      prevProps.layerProperties.equals(nextProps.layerProperties)
    );
  },
);

export default LayerListItem;
