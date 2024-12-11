import { memo, useCallback, MouseEvent } from "react";
import { ElementId } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";

// TODO: use some sort of unique ID instead of layerIndex/pointIndex,
// in order to support moving the point or layer in the hierarchy?

interface SkeletonListItemProps {
  documentManager: DocumentManager;
  id: ElementId;
  name: string;
  isHighlighted: boolean;
  isSelected: boolean;
}

export const SkeletonListItem = memo(function SkeletonListItem({
  documentManager,
  id,
  name,
  isHighlighted,
  isSelected,
}: SkeletonListItemProps) {
  const onMouseEnter = useCallback(() => {
    documentManager.setHighlightedElement(id);
  }, [documentManager, id]);

  const onMouseLeave = useCallback(() => {
    if (documentManager.highlightedElementId() === id) {
      documentManager.setHighlightedElement(undefined);
    }
  }, [documentManager, id]);

  const onSelectElement = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      if (event.shiftKey) {
        documentManager.toggleSelectedElement(id);
      } else {
        documentManager.setSelectedElements([id]);
      }
    },
    [documentManager, id],
  );

  let extraClass = "";
  if (isHighlighted) {
    extraClass += " is-highlighted";
  }
  if (isSelected) {
    extraClass += " is-selected";
  }

  return (
    <div
      className={`panel-list-item object-row-info${extraClass}`}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
    >
      <div className="highlight-zone" onClick={onSelectElement}>
        <p className="name single-line-text">{name}</p>
      </div>
    </div>
  );
});

export default SkeletonListItem;
