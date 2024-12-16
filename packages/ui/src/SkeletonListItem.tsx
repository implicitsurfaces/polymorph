import { memo, useCallback, MouseEvent } from "react";
import { ElementId } from "./Document.ts";
import { DocumentManager } from "./DocumentManager.ts";

// TODO: use some sort of unique ID instead of layerIndex/pointIndex,
// in order to support moving the point or layer in the hierarchy?

interface SkeletonListItemProps {
  documentManager: DocumentManager;
  id: ElementId;
  name: string;
  isHovered: boolean;
  isSelected: boolean;
}

export const SkeletonListItem = memo(function SkeletonListItem({
  documentManager,
  id,
  name,
  isHovered,
  isSelected,
}: SkeletonListItemProps) {
  const onMouseEnter = useCallback(() => {
    documentManager.setHoveredElement(id);
  }, [documentManager, id]);

  const onMouseLeave = useCallback(() => {
    if (documentManager.hoveredElementId() === id) {
      documentManager.setHoveredElement(undefined);
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
  if (isHovered) {
    extraClass += " is-hovered";
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
      <div className="hover-zone" onClick={onSelectElement}>
        <p className="name single-line-text">{name}</p>
      </div>
    </div>
  );
});

export default SkeletonListItem;
