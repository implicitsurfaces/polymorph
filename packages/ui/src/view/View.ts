import { CanvasSettings } from "../components/Canvas";

export interface View {
  readonly sideBySideCanvas: boolean;
  readonly leftCanvasSettings: CanvasSettings;
  readonly rightCanvasSettings: CanvasSettings;
}

export const defaultView: View = {
  sideBySideCanvas: true,
  leftCanvasSettings: new CanvasSettings(),
  rightCanvasSettings: new CanvasSettings({ sdfTest: true }),
};
