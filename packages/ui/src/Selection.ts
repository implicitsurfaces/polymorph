import { ElementId } from "./Document.ts";

export interface SelectableBase {
  readonly type: string;
}

export interface SelectableElement extends SelectableBase {
  readonly type: "Element";
  readonly id: ElementId;
}

export interface SelectableSubElement extends SelectableBase {
  readonly type: "SubElement";
  readonly id: ElementId;
  readonly subName: string;
}

export type Selectable = SelectableElement | SelectableSubElement;

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
    case "Element":
      if (s2.type !== "Element") {
        return false;
      }
      return s1.id === s2.id;
    case "SubElement":
      if (s2.type !== "SubElement") {
        return false;
      }
      return s1.id === s2.id && s1.subName === s2.subName;
  }
}

export class Selection {
  private _onChange: () => void;

  private _activeLayer: ElementId;
  private _hovered: Selectable | undefined;
  private _selected: Array<Selectable>;

  constructor(onChange: () => void) {
    this._onChange = onChange;
    this._activeLayer = "";
    this._hovered = undefined;
    this._selected = [];
  }

  activeLayer(): ElementId {
    return this._activeLayer;
  }

  setActiveLayer(id: ElementId) {
    if (this._activeLayer !== id) {
      this._activeLayer = id;
      this._onChange();
    }
  }

  hovered(): Selectable | undefined {
    return this._hovered;
  }

  hoveredElement(): ElementId | undefined {
    if (!this._hovered) {
      return undefined;
    }
    switch (this._hovered.type) {
      case "Element":
        return this._hovered.id;
      case "SubElement":
        return undefined;
    }
  }

  setHovered(selectable: Selectable | undefined) {
    if (!isSameSelectable(selectable, this._hovered)) {
      this._hovered = selectable;
      this._onChange();
    }
  }

  setHoveredElement(id: ElementId | undefined) {
    if (id) {
      this.setHovered({ type: "Element", id: id });
    } else {
      this.setHovered(undefined);
    }
  }

  isHovered(selectable: Selectable) {
    return isSameSelectable(selectable, this._hovered);
  }

  isHoveredElement(id: ElementId) {
    return this.isHovered({ type: "Element", id: id });
  }

  isHoveredSubElement(id: ElementId, subName: string) {
    return this.isHovered({
      type: "SubElement",
      id: id,
      subName: subName,
    });
  }

  selected(): Array<Selectable> {
    // XXX: Use Immutable.js instead?
    return [...this._selected];
  }

  selectedElements(): Array<ElementId> {
    const res: Array<ElementId> = [];
    for (const s of this._selected) {
      if (s.type === "Element") {
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

  setSelectedElements(ids: Array<ElementId>) {
    this._selected = ids.map((id) => {
      return { type: "Element", id: id };
    });
    this._onChange();
  }

  isSelected(selectable: Selectable) {
    const index = this._selected.findIndex((value: Selectable) => {
      return isSameSelectable(value, selectable);
    });
    return index !== -1;
  }

  isSelectedElement(id: ElementId) {
    return this.isSelected({ type: "Element", id: id });
  }

  isSelectedSubElement(id: ElementId, subName: string) {
    return this.isSelected({
      type: "SubElement",
      id: id,
      subName: subName,
    });
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

  toggleSelectedElement(id: ElementId) {
    this.toggleSelected({ type: "Element", id: id });
  }
}
