import { memo, useCallback } from "react";
import { DocumentManager, Point, ElementId } from "./Document.ts";
import { NumberInput } from "./NumberInput.tsx";

// TODO: use some sort of unique ID instead of layerIndex/pointIndex,
// in order to support moving the point or layer in the hierarchy?

interface SkeletonListItemProps {
  documentManager: DocumentManager;
  id: ElementId;
  point: Point; // we need this for memoization
}

export const SkeletonListItem = memo(
  function SkeletonListItem({
    documentManager,
    id,
    point,
  }: SkeletonListItemProps) {
    const onXChange = useCallback(
      (value: number) => {
        const point = documentManager.document().getElementFromId<Point>(id);
        if (point) {
          point.position.x = value;
          documentManager.commitChanges();
        }
      },
      [documentManager, id],
    );

    const onYChange = useCallback(
      (value: number) => {
        const point = documentManager.document().getElementFromId<Point>(id);
        if (point) {
          point.position.y = value;
          documentManager.commitChanges();
        }
      },
      [documentManager, id],
    );

    return (
      <div className="panel-list-item object-row-info">
        <div className="highlight-zone">
          <p className="name single-line-text">{point.name}</p>
        </div>
        <div className="extra-zone">
          <NumberInput
            idBase={`number-input::x${id}`}
            label="X"
            value={point.position.x}
            onChange={onXChange}
          />
          <NumberInput
            idBase={`number-input::y${id}`}
            label="Y"
            value={point.position.y}
            onChange={onYChange}
          />
        </div>
      </div>
    );
  },
  (prevProps, nextProps) => {
    // Avoid re-rendering if all the props are the same.
    //
    // For points, we need value equality rather than reference equality
    // because:
    //
    // - We perform deep copies on undo/redo/commit, so the point references might
    //   differ but their value are still equal and there is no need to re-render.
    //
    // - We do not perform any copy on stage (e.g., dragging a point), so the point
    //   references might be equal but their value differ, and we need to re-render
    //   in this case.
    //
    // In the future, this design may change, for example if we decide to use
    // Immer-like shallow copies (or Immutable.js-like structural sharing) instead of
    // deep copies, and if we also decide to not mutate state even on stage.
    //
    return (
      prevProps.documentManager === nextProps.documentManager &&
      prevProps.id === nextProps.id &&
      prevProps.point.equals(nextProps.point)
    );
  },
);

export default SkeletonListItem;
