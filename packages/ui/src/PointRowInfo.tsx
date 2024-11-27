import { memo, useCallback } from 'react';
import { DocumentManager, Point } from './Document.ts';
import { NumberInput } from './NumberInput.tsx';

// TODO: use some sort of unique ID instead of layerIndex/pointIndex,
// in order to support moving the point or layer in the hierarchy?

interface PointRowInfoProps {
  documentManager: DocumentManager;
  layerIndex: number;
  pointIndex: number;
  point: Point; // we need this for memoization
}

export const PointRowInfo = memo(
  function PointRowInfo({ documentManager, layerIndex, pointIndex, point }: PointRowInfoProps) {
    const onXChange = useCallback(
      (value: number) => {
        documentManager.document().layers[layerIndex].points[pointIndex].position.x = value;
        documentManager.commitChanges();
      },
      [documentManager, layerIndex, pointIndex]
    );

    const onYChange = useCallback(
      (value: number) => {
        documentManager.document().layers[layerIndex].points[pointIndex].position.y = value;
        documentManager.commitChanges();
      },
      [documentManager, layerIndex, pointIndex]
    );

    return (
      <div className="panel-list-item object-row-info">
        <p className="name">{point.name}</p>
        <NumberInput
          idBase={`number-input::x${pointIndex}`}
          label="X"
          value={point.position.x}
          onChange={onXChange}
        />
        <NumberInput
          idBase={`number-input::y${pointIndex}`}
          label="Y"
          value={point.position.y}
          onChange={onYChange}
        />
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
      prevProps.layerIndex === nextProps.layerIndex &&
      prevProps.pointIndex === nextProps.pointIndex &&
      prevProps.point.equals(nextProps.point)
    );
  }
);

export default PointRowInfo;
