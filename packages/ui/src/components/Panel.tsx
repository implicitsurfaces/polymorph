export { Panel, PanelGroup } from "react-resizable-panels";

import {
  PanelResizeHandle as _PanelResizeHandle,
  PointerHitAreaMargins,
} from "react-resizable-panels";

import "./Panel.css";

function panelHitMargins(): PointerHitAreaMargins {
  // separator (0-2px) + 2 * margins (3px) = 6-8px total hit area
  return { coarse: 3, fine: 3 };
}

export function PanelResizeHandle() {
  return <_PanelResizeHandle hitAreaMargins={panelHitMargins()} />;
}
