import { ElementId } from "./Document.ts";

export class Selection {
  private _onChange: () => void;

  private _activeLayerId: ElementId;
  private _hoveredElementId: ElementId | undefined;
  private _selectedElementIds: Array<ElementId>;

  constructor(onChange: () => void) {
    this._onChange = onChange;
    this._activeLayerId = "";
    this._hoveredElementId = undefined;
    this._selectedElementIds = [];
  }

  activeLayerId(): ElementId {
    return this._activeLayerId;
  }

  setActiveLayer(id: ElementId) {
    if (this._activeLayerId !== id) {
      this._activeLayerId = id;
      this._onChange();
    }
  }

  hoveredElementId(): ElementId | undefined {
    return this._hoveredElementId;
  }

  setHoveredElement(id: ElementId | undefined) {
    if (this._hoveredElementId !== id) {
      this._hoveredElementId = id;
      this._onChange();
    }
  }

  selectedElementIds(): Array<ElementId> {
    // XXX: Use Immutable.js instead?
    return [...this._selectedElementIds];
  }

  setSelectedElements(ids: Array<ElementId>) {
    // XXX: Use Immutable.js instead?
    this._selectedElementIds = [...ids];
    this._onChange();
  }

  toggleSelectedElement(id: ElementId) {
    const selectedIds = this.selectedElementIds();
    const index = selectedIds.indexOf(id);
    if (index === -1) {
      selectedIds.push(id);
    } else {
      selectedIds.splice(index, 1);
    }
    this.setSelectedElements(selectedIds);
  }
}
