import { Point, DocumentManager } from './Document.ts';
import { PointRowInfo } from './PointRowInfo.tsx';

interface ObjectsPanelProps {
  documentManager: DocumentManager;
}

export function ObjectsPanel({ documentManager }: ObjectsPanelProps) {
  const layerIndex = documentManager.activeLayerIndex();
  const layer = documentManager.activeLayer();

  let panelBody;
  if (layer) {
    panelBody = layer.points.map((point: Point, pointIndex: number) => (
      <PointRowInfo
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
      <h2 className="panel-title">Objects</h2>
      <div className="panel-body">{panelBody}</div>
    </div>
  );
}

export default ObjectsPanel;
