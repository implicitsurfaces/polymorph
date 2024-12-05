import { useCallback, ChangeEvent, memo } from "react";

interface NumberInputProps {
  idBase: string;
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
    <div id={idBase} data-scope="number-input" data-part="root" dir="ltr">
      <label
        data-scope="number-input"
        data-part="label"
        dir="ltr"
        id={`${idBase}::label`}
        htmlFor={`${idBase}::input`}
      >
        {label}
      </label>
      <input
        data-scope="number-input"
        data-part="input"
        dir="ltr"
        id={`${idBase}::input`}
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
