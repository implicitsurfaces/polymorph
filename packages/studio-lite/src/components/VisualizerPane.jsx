import { useRef, forwardRef } from "react";
import { styled } from "goober";
import { observer } from "mobx-react";

import Splitter, { GutterTheme, SplitDirection } from "@devbookhq/splitter";

import { HeaderSelect, Spacer } from "./panes";

import useEditorStore from "../state/useEditorStore";

const Canvas = styled("canvas", forwardRef)`
  max-width: 100%;
  aspect-ratio: 1;
  margin: 0 auto auto 0;
`;

const Image = observer(() => {
  const store = useEditorStore();
  const canvasRef = useRef(null);

  console.log("Rendering VisualizerPane");

  if (store.currentImage && canvasRef?.current) {
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

const ValueReadWrapper = styled("div")`
  display: flex;
  flex-direction: column;
  width: 7rem;
  font-size: 0.8em;

  & > :first-child {
    font-weight: bold;
    width: 100%;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
  }
`;

const fmt = (v) => {
  return Intl.NumberFormat("en", { maximumSignificantDigits: 3 }).format(v);
};

function ValueRead({ name, value }) {
  if (value === null) return null;
  const valueTxt = Array.isArray(value)
    ? `(${value.map(fmt).join(", ")})`
    : fmt(value);
  return (
    <ValueReadWrapper>
      <span>{name}</span>
      <span>{valueTxt}</span>
    </ValueReadWrapper>
  );
}

const ValueReadsWrapper = styled("div")`
  overflow-y: auto;
  width: 100%;
  display: flex;
  flex-wrap: wrap;
  gap: 0.4em;
  padding: 0.9em;
`;

const SplitterContainer = styled("div")`
  display: flex;
  width: 100%;
  height: 100%;
  overflow: hidden;
`;

export const VisualizerPane = observer(() => {
  const store = useEditorStore();
  return (
    <Splitter
      direction={SplitDirection.Vertical}
      gutterTheme={GutterTheme.Dark}
      gutterClassName="custom-gutter-theme"
      initialSizes={store.valueReads.length ? [75, 25] : [100]}
    >
      <SplitterContainer>
        <Image />
      </SplitterContainer>
      {!!store.valueReads.length && (
        <ValueReadsWrapper>
          {store.valueReads.map((value, index) => (
            <ValueRead key={index} {...value} />
          ))}
        </ValueReadsWrapper>
      )}
    </Splitter>
  );
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
