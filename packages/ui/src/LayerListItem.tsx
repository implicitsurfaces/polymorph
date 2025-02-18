import { memo, useCallback, MouseEvent } from "react";
import { NodeId } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";

interface LayerListItemProps {
  documentManager: DocumentManager;
  id: NodeId;
  index: number;
  name: string;
  isActive: boolean;
}

export const LayerListItem = memo(function LayerListItem({
  documentManager,
  id,
  index,
  name,
  isActive,
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
    documentManager.selection().setActiveLayerId(id);
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
      <div className="hover-zone" onClick={onSelectLayer}>
        <p className="name single-line-text">{name}</p>
      </div>
    </div>
  );
});

export default LayerListItem;
