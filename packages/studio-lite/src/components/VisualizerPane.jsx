import React, { useRef } from "react";
import { styled } from "goober";
import { observer } from "mobx-react";

import { HeaderSelect, Spacer } from "./panes";

import useEditorStore from "../state/useEditorStore";

const Canvas = styled("canvas", React.forwardRef)`
  width: 100%;
  aspect-ratio: 1;
  margin: auto;
`;

export const VisualizerPane = observer(() => {
  const store = useEditorStore();
  const canvasRef = useRef(null);

  if (canvasRef?.current && store.currentImage) {
    canvasRef.current.width = store.currentDefinition;
    canvasRef.current.height = store.currentDefinition;
    canvasRef.current
      .getContext("2d")
      ?.putImageData(
        new ImageData(
          store.currentImage,
          store.currentDefinition,
          store.currentDefinition,
        ),
        0,
        0,
      );
  }

  return <Canvas ref={canvasRef}></Canvas>;
});

const DEFINITONS = [
  { name: "Very Low", value: 250 },
  { name: "Low", value: 750 },
  { name: "Medium", value: 1500 },
  { name: "High", value: 3000 },
];

export const VisualizerButtons = observer(() => {
  const store = useEditorStore();

  return (
    <>
      <>
        <HeaderSelect
          value={store.definition}
          onChange={(e) => store.changeDefinition(parseInt(e.target.value))}
        >
          {DEFINITONS.map(({ name, value }) => (
            <option value={value} key={value}>
              {name}
            </option>
          ))}
        </HeaderSelect>
        <Spacer />
      </>
    </>
  );
});
