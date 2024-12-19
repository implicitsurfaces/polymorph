import React from "react";
import { styled } from "goober";
import { observer } from "mobx-react";
import Editor from "@monaco-editor/react";

import Splitter, { GutterTheme, SplitDirection } from "@devbookhq/splitter";
import { HeaderButton, Spacer } from "./panes";
import { TreeUI } from "./TreeUI";

import LinkCode from "../icons/LinkCode";

import drawAPITypes from "draw-api/dist/main.d.ts?raw";

import "../loadMonaco";
import useEditorStore from "../state/useEditorStore";
import { exportCode } from "../state/codeInit";
import LoadingScreen from "./LoadingScreen";

import { Dialog, DialogTitle } from "./Dialog";

export const ErrorOverlay = styled("div")`
  display: flex;
  flex-direction: column;
  min-width: 100%;
  gap: 0.4em;
  padding: 2em;
  border-color: red;
  border-width: 2px;
  max-height: initial;
  max-width: 50vw;
  max-height: 90vw;

  & > :first-child {
    color: red;
  }

  & > :nth-child(2) {
    font-size: 1.2em;
  }

  & > pre {
    font-size: 0.6em;
    overflow-x: auto;
    padding: 1em;
    background-color: #f2e0de;
  }
`;

export const InfoOverlay = styled("div")`
  padding: 2em;
  font-size: 0.7em;
  position: absolute;
  height: 100px;
  width: 100%;
  bottom: 0;
  background-color: #f2e0de;
`;

const DRAW_API = `
  import * as modAll from 'drawAPImod';
  declare global {
  import * as modAll from 'drawAPImod';
      declare const drawAPI = modAll;
  }
`;

export const EditorPane = observer(function EditorPane() {
  const store = useEditorStore();

  const handleEditorDidMount = (_, monaco) => {
    monaco.languages.typescript.javascriptDefaults.setEagerModelSync(true);

    monaco.languages.typescript.javascriptDefaults.setExtraLibs([
      {
        content: `declare module 'drawAPImod' { ${drawAPITypes} }`,
      },
      {
        content: DRAW_API,
      },
    ]);
  };

  if (!store.code.initialized) return <LoadingScreen />;

  return (
    <>
      <Splitter
        direction={SplitDirection.Vertical}
        gutterTheme={GutterTheme.Dark}
        gutterClassName="custom-gutter-theme"
        initialSizes={store.error ? [75, 25] : [100]}
      >
        <Editor
          defaultLanguage="javascript"
          defaultValue={store.code.current}
          theme="vs-dark"
          height="100%"
          onChange={(e) => {
            store.code.update(e, true);
          }}
          onMount={handleEditorDidMount}
          options={{
            automaticLayout: true,
            minimap: { enabled: false },
          }}
        />
        {store.error && (
          <ErrorOverlay>
            <div>Error</div>
            <div>{store.error?.message}</div>
            {store.error.stack && <pre>{store.error.stack}</pre>}
          </ErrorOverlay>
        )}
      </Splitter>
    </>
  );
});

export const EditorButtons = observer(() => {
  const store = useEditorStore();
  const [linkShare, setLinkShare] = React.useState(false);
  const [tree, setTree] = React.useState(false);

  const share = async () => {
    setLinkShare(true);
    const url = await exportCode(store.code.current);
    navigator.clipboard.writeText(url);
    setTimeout(() => {
      setLinkShare(false);
    }, 1000);
  };

  const showTree = () => {
    setTree(true);
  };

  const style = linkShare ? { color: "lightgreen" } : {};

  return (
    <>
      <HeaderButton onClick={share} title="Link to current code">
        <LinkCode style={style} />
      </HeaderButton>
      <HeaderButton onClick={showTree} title="Link to current code">
        Tree
      </HeaderButton>
      <Spacer />
      {tree && (
        <Dialog onClose={() => setTree(false)}>
          <DialogTitle onClose={() => setTree(false)}>
            Num tree explorator
          </DialogTitle>
          <TreeUI />
        </Dialog>
      )}
    </>
  );
});
