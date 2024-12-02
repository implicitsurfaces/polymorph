import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './reset.css';
import App from './App.tsx';

import { Repo, isValidAutomergeUrl } from '@automerge/automerge-repo';
import { BroadcastChannelNetworkAdapter } from '@automerge/automerge-repo-network-broadcastchannel';
//import { BrowserWebSocketClientAdapter } from '@automerge/automerge-repo-network-websocket';
import { IndexedDBStorageAdapter } from '@automerge/automerge-repo-storage-indexeddb';
import { RepoContext } from '@automerge/automerge-repo-react-hooks';

import { ADocument } from './Document.ts';

const repo = new Repo({
  network: [
    //new BrowserWebSocketClientAdapter('wss://sync.automerge.org'),
    new BroadcastChannelNetworkAdapter(),
  ],
  storage: new IndexedDBStorageAdapter(),
});

const rootDocUrl = document.location.hash.substring(1);

let handle;
if (isValidAutomergeUrl(rootDocUrl)) {
  handle = repo.find(rootDocUrl);
} else {
  handle = repo.create<ADocument>({
    layers: [
      {
        properties: {
          name: 'Layer 1',
        },
        points: [],
      },
    ],
    activeLayerIndex: 0,
  });
}

document.location.hash = handle.url;

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RepoContext.Provider value={repo}>
      <App docUrl={handle.url} />
    </RepoContext.Provider>
  </StrictMode>
);
