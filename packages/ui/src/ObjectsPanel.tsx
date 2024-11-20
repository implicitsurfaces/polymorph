import { Point, DocumentManager } from './Document.ts';
import { PointRowInfo } from './PointRowInfo.tsx';

import './Panel.css';
import './ObjectsPanel.css';

interface ObjectsPanelProps {
  documentManager: DocumentManager;
}

export function ObjectsPanel({ documentManager }: ObjectsPanelProps) {
  return (
    <div className="panel">
      <h2 className="panel-title">Objects</h2>
      <div className="panel-body">
        {documentManager.document().points.map((point: Point, index: number) => (
          <PointRowInfo key={index} documentManager={documentManager} index={index} point={point.clone()} />
        ))}
      </div>
    </div>
  );
}

export default ObjectsPanel;
