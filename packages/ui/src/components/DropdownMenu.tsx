import { ReactNode } from "react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";

import { useDocumentManager } from "../doc/DocumentManagerContext";
import { Action, TriggerAction } from "../actions/Action";

import "./DropdownMenu.css";
import { useViewContext } from "../view";

interface MenuProps {
  children: ReactNode;
  trigger: ReactNode;
}

export function Menu({ children, trigger = "Menu" }: MenuProps) {
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger>{trigger}</DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          className="dropdown-menu"
          alignOffset={-8}
          align="start"
          sideOffset={10}
        >
          {children}
          <DropdownMenu.Arrow className="arrow" />
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

interface MenuActionItemProps {
  action: Action;
}

export function MenuActionItem({ action }: MenuActionItemProps) {
  const documentManager = useDocumentManager();
  const viewContext = useViewContext();

  function onClick(action: Action) {
    if (action instanceof TriggerAction) {
      action.onTrigger(documentManager, viewContext);
    }
  }

  return (
    <DropdownMenu.Item key={action.name} onClick={() => onClick(action)}>
      <span>{action.name}</span>
      {action.shortcut ? (
        <span className="right-slot">{action.shortcut.prettyStr}</span>
      ) : undefined}
    </DropdownMenu.Item>
  );
}
