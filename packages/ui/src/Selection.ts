import { NodeId } from "./Document.ts";

export interface SelectableBase {
  readonly type: string;
}

export interface SelectableNode extends SelectableBase {
  readonly type: "Node";
  readonly id: NodeId;
}

export type Selectable = SelectableNode;

function isSameSelectable(
  s1: Selectable | undefined,
  s2: Selectable | undefined,
) {
  if (!s1) {
    return s2 === undefined;
  }
  if (!s2) {
    return s1 === undefined;
  }
  switch (s1.type) {
    case "Node":
      if (s2.type !== "Node") {
        return false;
      }
      return s1.id === s2.id;
  }
}

export class Selection {
  private _onChange: () => void;

  private _activeLayer: NodeId;
  private _hovered: Selectable | undefined;
  private _selected: Array<Selectable>;

  constructor(onChange: () => void) {
    this._onChange = onChange;
    this._activeLayer = "";
    this._hovered = undefined;
    this._selected = [];
  }

  activeLayer(): NodeId {
    return this._activeLayer;
  }

  setActiveLayer(id: NodeId) {
    if (this._activeLayer !== id) {
      this._activeLayer = id;
      this._onChange();
    }
  }

  hovered(): Selectable | undefined {
    return this._hovered;
  }

  hoveredNode(): NodeId | undefined {
    if (!this._hovered) {
      return undefined;
    }
    switch (this._hovered.type) {
      case "Node":
        return this._hovered.id;
    }
  }

  setHovered(selectable: Selectable | undefined) {
    if (!isSameSelectable(selectable, this._hovered)) {
      this._hovered = selectable;
      this._onChange();
    }
  }

  setHoveredNode(id: NodeId | undefined) {
    if (id) {
      this.setHovered({ type: "Node", id: id });
    } else {
      this.setHovered(undefined);
    }
  }

  isHovered(selectable: Selectable) {
    return isSameSelectable(selectable, this._hovered);
  }

  isHoveredNode(id: NodeId) {
    return this.isHovered({ type: "Node", id: id });
  }

  selected(): Array<Selectable> {
    // XXX: Use Immutable.js instead?
    return [...this._selected];
  }

  selectedNodes(): Array<NodeId> {
    const res: Array<NodeId> = [];
    for (const s of this._selected) {
      if (s.type === "Node") {
        res.push(s.id);
      }
    }
    return res;
  }

  setSelected(selectables: Array<Selectable>) {
    // XXX: Use Immutable.js instead?
    this._selected = [...selectables];
    this._onChange();
  }

  setSelectedNodes(ids: Array<NodeId>) {
    this._selected = ids.map((id) => {
      return { type: "Node", id: id };
    });
    this._onChange();
  }

  isSelected(selectable: Selectable) {
    const index = this._selected.findIndex((value: Selectable) => {
      return isSameSelectable(value, selectable);
    });
    return index !== -1;
  }

  isSelectedNode(id: NodeId) {
    return this.isSelected({ type: "Node", id: id });
  }

  toggleSelected(selectable: Selectable) {
    const selected = this.selected();
    const index = selected.findIndex((value: Selectable) => {
      return isSameSelectable(value, selectable);
    });
    if (index === -1) {
      selected.push(selectable);
    } else {
      selected.splice(index, 1);
    }
    this.setSelected(selected);
  }

  toggleSelectedNode(id: NodeId) {
    this.toggleSelected({ type: "Node", id: id });
  }
}
