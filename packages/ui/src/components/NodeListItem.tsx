import { memo, useCallback, MouseEvent } from "react";
import { NodeId } from "../doc/Node";
import { DocumentManager } from "../doc/DocumentManager";

interface NodeListItemProps {
  documentManager: DocumentManager;
  id: NodeId;
  name: string;
  isHovered: boolean;
  isSelected: boolean;
  title?: string;
}

export const NodeListItem = memo(function NodeListItem({
  documentManager,
  id,
  name,
  isHovered,
  isSelected,
  title,
}: NodeListItemProps) {
  const onMouseEnter = useCallback(() => {
    documentManager.selection().setHoveredNodeId(id);
  }, [documentManager, id]);

  const onMouseLeave = useCallback(() => {
    if (documentManager.selection().hoveredNodeId() === id) {
      documentManager.selection().setHoveredNodeId(undefined);
    }
  }, [documentManager, id]);

  const onSelectNode = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      if (event.shiftKey) {
        documentManager.selection().toggleSelectedNodeId(id);
      } else {
        documentManager.selection().setSelectedNodeIds([id]);
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
      {title && (
        <div className="extra-zone">
          <p className="name single-line-text">{title}</p>
        </div>
      )}
      <div className="hover-zone" onClick={onSelectNode}>
        <p className="name single-line-text">{name}</p>
      </div>
    </div>
  );
});

export default NodeListItem;
