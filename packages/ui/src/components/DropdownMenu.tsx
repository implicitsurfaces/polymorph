import { ReactNode, useContext } from "react";
import * as DropdownMenu from "@radix-ui/react-dropdown-menu";

import { DocumentManagerContext } from "../doc/DocumentManagerContext";
import { Action, TriggerAction } from "../actions/Action";

import "./DropdownMenu.css";

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

interface ActionItemProps {
  action: Action;
}

export function ActionItem({ action }: ActionItemProps) {
  const { documentManager } = useContext(DocumentManagerContext);

  function onClick(action: Action) {
    if (action instanceof TriggerAction) {
      action.onTrigger(documentManager);
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
