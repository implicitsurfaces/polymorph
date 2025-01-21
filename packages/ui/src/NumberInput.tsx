import { v4 as uuidv4 } from "uuid";
import { useState, useCallback, ChangeEvent, memo } from "react";

interface NumberInputProps {
  idBase?: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
}

export const NumberInput = memo(function NumberInput({
  idBase,
  label,
  value,
  onChange,
}: NumberInputProps) {
  // Automatically create a unique idBase if not explicitly provided.
  //
  const [_idBase] = useState<string>(() => {
    if (idBase) {
      return idBase;
    } else {
      return uuidv4();
    }
  });

  const _onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      let newValue = parseFloat(event.target.value);
      if (isNaN(newValue)) {
        newValue = 0;
      }
      onChange(newValue);
    },
    [onChange],
  );

  return (
    <div id={_idBase} data-scope="number-input" data-part="root" dir="ltr">
      <label
        data-scope="number-input"
        data-part="label"
        dir="ltr"
        id={`${_idBase}::label`}
        htmlFor={`${_idBase}::input`}
      >
        {label}
      </label>
      <input
        data-scope="number-input"
        data-part="input"
        dir="ltr"
        id={`${_idBase}::input`}
        role="spinbutton"
        pattern="[0-9]*(.[0-9]+)?"
        inputMode="decimal"
        autoComplete="off"
        autoCorrect="off"
        spellCheck="false"
        type="text"
        aria-roledescription="numberfield"
        aria-valuemin={-9007199254740991}
        aria-valuemax={9007199254740991}
        aria-valuenow={value}
        value={value}
        onChange={_onChange}
      />
    </div>
  );
});

export default NumberInput;
