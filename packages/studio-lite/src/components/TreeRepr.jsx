import { memo, useState, useRef, useEffect, forwardRef } from "react";
import { parseDOTNetwork } from "vis-network";
import cytoscape from "cytoscape";
import dagre from "cytoscape-dagre";
import { DialogTitle } from "./Dialog";

import { styled } from "goober";

cytoscape.use(dagre);

const Tree = styled("div", forwardRef)`
  width: 90vw;
  height: 90vh;
`;

export const TreeRepr = memo(function TreeRepr({ tree }) {
  const [container, setContainer] = useState(null);

  const currentTree = useRef(null);
  const currentNetwork = useRef(null);

  useEffect(() => {
    if (tree === currentTree.current || !container) {
      return;
    }

    const data = parseDOTNetwork(tree);

    const elements = [
      ...data.nodes.map((node) => ({
        data: {
          id: node.id,
          label: node.label,
        },
      })),
      ...data.edges.map((edge) => ({
        data: {
          source: edge.from,
          target: edge.to,
          id: edge.id,
        },
      })),
    ];
    if (currentNetwork.current) {
      currentNetwork.current.setData(data);
      currentNetwork.current.resize();
    } else {
      currentNetwork.current = cytoscape({
        container,
        elements,
        layout: { name: "dagre", fit: true, align: "UL" },
        style: [
          {
            selector: "node[label]",
            style: {
              label: "data(label)",
            },
          },
          { selector: ".collapsed", style: { display: "none" } },
        ],
      });
      currentNetwork.current.resize();
      currentNetwork.current.on("tap", "node", function (evt) {
        evt.target.successors().toggleClass("collapsed");
      });

      currentNetwork.current.on("tap", "edge", function (evt) {
        currentNetwork.current.pan(evt.target.source().position());
      });
    }

    currentTree.current = tree;
  }, [tree, container]);

  return (
    <>
      <Tree ref={setContainer} />
    </>
  );
});
