import { Point, DocumentManager } from './Document.ts';
import { SkeletonListItem } from './SkeletonListItem.tsx';

interface SkeletonPanelProps {
  documentManager: DocumentManager;
}

export function SkeletonPanel({ documentManager }: SkeletonPanelProps) {
  const layerIndex = documentManager.activeLayerIndex();
  const layer = documentManager.activeLayer();

  let panelBody;
  if (layer) {
    panelBody = layer.points.map((point: Point, pointIndex: number) => (
      <SkeletonListItem
        key={pointIndex}
        documentManager={documentManager}
        layerIndex={layerIndex}
        pointIndex={pointIndex}
        point={point.clone()}
      />
    ));
  } else {
    panelBody = <></>;
  }

  return (
    <div className="panel">
      <h2 className="panel-title">Skeleton</h2>
      <div className="panel-body">{panelBody}</div>
    </div>
  );
}

export default SkeletonPanel;
