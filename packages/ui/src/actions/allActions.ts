import { Action } from "./Action.ts";

import { AddDistanceAction } from "./AddDistanceAction.ts";

export function allActions(): Action[] {
  return [new AddDistanceAction()];
}
