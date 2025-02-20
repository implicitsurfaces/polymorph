import { Vector2 } from "threejs-math";
import { useCallback } from "react";
import { NumberInput } from "./NumberInput";

// Note: using a getter function for position (instead of a value)
// makes it possible to memoize the onXChange and onYChange functions.
// Otherwise, the onXChange function would have to capture to "currentY"
// value, meaning it would have to change each time y changes.
//
interface Vector2InputProps {
  getValue: () => Vector2;
  setValue: (v: Vector2) => void;
}

export function Vector2Input({ getValue, setValue }: Vector2InputProps) {
  const position = getValue();

  const onXChange = useCallback(
    (value: number) => {
      const newPos = getValue().clone().setX(value);
      setValue(newPos);
    },
    [getValue, setValue],
  );

  const onYChange = useCallback(
    (value: number) => {
      const newPos = getValue().clone().setY(value);
      setValue(newPos);
    },
    [getValue, setValue],
  );

  return (
    <>
      <NumberInput label="x" value={position.x} onChange={onXChange} />
      <NumberInput label="y" value={position.y} onChange={onYChange} />
    </>
  );
}
