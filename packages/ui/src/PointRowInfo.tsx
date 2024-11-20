import { memo, useCallback } from 'react';
import { DocumentManager, Point } from './Document.ts';
import { NumberInput } from './NumberInput.tsx';

import './Panel.css';
import './ObjectsPanel.css';

interface PointRowInfoProps {
  documentManager: DocumentManager;
  index: number; // TODO: use some sort of unique ID instead? (e.g., if moved in hierarchy)
  point: Point; // we need this for memoization
}

export const PointRowInfo = memo(
  function PointRowInfo({ documentManager, index, point }: PointRowInfoProps) {
    const onXChange = useCallback(
      (value: number) => {
        documentManager.document().points[index].position.x = value;
        documentManager.commitChanges();
      },
      [documentManager, index]
    );

    const onYChange = useCallback(
      (value: number) => {
        documentManager.document().points[index].position.y = value;
        documentManager.commitChanges();
      },
      [documentManager, index]
    );

    return (
      <div className="object-row-info">
        <p className="object-name">{point.name}</p>
        <NumberInput
          idBase={`number-input::x${index}`}
          label="X"
          value={point.position.x}
          onChange={onXChange}
        />
        <NumberInput
          idBase={`number-input::y${index}`}
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
      prevProps.index === nextProps.index &&
      prevProps.point.equals(nextProps.point)
    );
  }
);

export default PointRowInfo;
