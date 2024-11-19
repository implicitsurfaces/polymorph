import { useCallback, ChangeEvent } from 'react';

interface NumberInputProps {
  idBase: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
}

export function NumberInput({ idBase, label, value, onChange }: NumberInputProps) {
  const _onChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const newValue = parseFloat(event.target.value);
      onChange(newValue);
    },
    [onChange]
  );

  return (
    <div id={idBase} data-scope="number-input" data-part="root" dir="ltr">
      <div
        data-scope="number-input"
        data-part="scrubber"
        dir="ltr"
        id={`${idBase}::scrubber`}
        role="presentation"
      >
        <label
          data-scope="number-input"
          data-part="label"
          dir="ltr"
          id={`${idBase}::label`}
          htmlFor={`${idBase}::input`}
        >
          {label}
        </label>
      </div>
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
}

export default NumberInput;
