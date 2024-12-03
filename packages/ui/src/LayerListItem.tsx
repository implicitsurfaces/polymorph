import { memo, useCallback, MouseEvent } from 'react';
import { DocumentManager } from './Document.ts';
import { ALayerProperties, ADocument } from './Document.ts';

import type { AutomergeUrl } from '@automerge/automerge-repo';
import { useDocument } from '@automerge/automerge-repo-react-hooks';

interface LayerListItemProps {
  documentManager: DocumentManager;
  docUrl: AutomergeUrl;
  index: number; // TODO: use some sort of unique ID instead? (e.g., if moved in hierarchy)
  isActive: boolean;
  layerProperties: ALayerProperties; // we need this for memoization
}

export const LayerListItem = memo(
  function LayerListItem({ documentManager, docUrl, index, isActive, layerProperties }: LayerListItemProps) {
    const [doc, changeDoc] = useDocument<ADocument>(docUrl);

    const onCreateLayer = useCallback(
      (event: MouseEvent<HTMLButtonElement>) => {
        // Click: insert after
        // Alt+Click: insert before
        if (!doc) {
          return;
        }
        const insertIndex = event.altKey ? index : index + 1;
        changeDoc(d => {
          const name = 'Layer ' + (doc.layers.length + 1);
          const prevActiveLayer = d.activeLayerIndex;
          let nextActiveLayer = prevActiveLayer;
          if (insertIndex <= prevActiveLayer) {
            nextActiveLayer += 1;
          }
          // XXX: Why isn't insertAt from Automerge available?
          d.layers.splice(insertIndex, 0, { properties: { name: name }, points: [] });
          d.activeLayerIndex = nextActiveLayer;
        });
      },
      [doc, changeDoc, index]
    );

    const onDeleteLayer = useCallback(() => {
      // Prevent deleting the last layer: this is important with the current
      // design (+/- buttons next to each layer), otherwise after the last
      // layer is deleted, it is impossible to create layers.
      if (!doc || doc.layers.length == 1) {
        return;
      }

      changeDoc(d => {
        const prevActiveLayer = d.activeLayerIndex;
        let nextActiveLayer = prevActiveLayer;
        if (index < prevActiveLayer) {
          nextActiveLayer -= 1;
        }
        // XXX: Why isn't deleteAt from Automerge available?
        d.layers.splice(index, 1);
        d.activeLayerIndex = nextActiveLayer;
      });
    }, [documentManager, index]);

    const onSelectLayer = useCallback(() => {
      documentManager.setActiveLayer(index);
    }, [doc, changeDoc, index]);

    return (
      <div className={'panel-list-item has-secret-zone' + (isActive ? ' is-active' : '')}>
        <div className="secret-zone">
          <button className="single-character" onClick={onDeleteLayer}>
            -
          </button>
          <button className="single-character" onClick={onCreateLayer}>
            +
          </button>
        </div>
        <div className="highlight-zone" onClick={onSelectLayer}>
          <p className="name single-line-text">{layerProperties.name}</p>
        </div>
      </div>
    );
  },
  (prevProps, nextProps) => {
    // We need re-rendering only if the layer's properties (name, color, etc.)
    // change, but not if the layer's inner objects change.
    //
    /*
    return (
      prevProps.documentManager === nextProps.documentManager &&
      prevProps.index === nextProps.index &&
      prevProps.isActive === nextProps.isActive &&
      prevProps.layerProperties === prevProps.layerProperties
    );
    */
    return false;
  }
);

export default LayerListItem;
