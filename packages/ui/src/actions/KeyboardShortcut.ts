export function isMacPlatform() {
  // Note: `navigator.platform` is deprecated. An alternative might be to use
  // `navigator.userAgent.includes('Mac')`, but it feels more
  // correct/specific to use `platform` rather than `userAgent`.
  return (
    navigator.platform.indexOf("Mac") === 0 || navigator.platform === "iPhone"
  );
}

/**
 * Represents a keyboard shortcut.
 */
export class KeyboardShortcut {
  readonly key: string;
  readonly ctrlKey: boolean;
  readonly shiftKey: boolean;
  readonly altKey: boolean;
  readonly metaKey: boolean;
  readonly str: string;
  readonly prettyStr: string;

  /**
   * Creates a keyboard shortcut from a string.
   *
   * Example:
   *
   * ```
   * const copyShortcut = new Shortcut("CtrlCmd+C");
   * ```
   *
   * The input must be of the form "{mod1}+{mod2}+...+{key}", where allowed
   * modifiers are:
   *
   * - "Ctrl"  (Ctrl on Windows, Control on Mac)
   * - "Alt"   (Alt on Windows, Option on Mac)
   * - "Shift" (Shift on Windows, Shift on Mac)
   * - "Meta"  (Win on Windows, Command on Mac)
   *
   * - "CtrlCmd" (Ctrl on Windows, Command on Mac)
   * - "WinCtrl" (Win on Windows, Control on Mac)
   */
  constructor(str: string) {
    this.key = "";
    this.ctrlKey = false;
    this.altKey = false;
    this.shiftKey = false;
    this.metaKey = false;
    this.str = "";
    this.prettyStr = "";

    // Remove all whitespace characters
    str = str.replace(/\s/g, "");

    // Extract the key and modifiers.
    //
    // If the string is empty, there is no key nor modifiers (= no shortcut).
    // If the string ends in `+`, then the key itself is assumed to be `+`.
    //
    // We implement this by splitting the string at "+" characters. Examples:
    // ""       => [""]
    // "C"      => ["C"]
    // "Ctrl+C" => ["Ctrl", "C"]
    // "+"      => ["", ""]
    // "Ctrl+"  => ["Ctrl", ""]
    // "Ctrl++" => ["Ctrl", "", ""]
    //
    const tokens = str.split("+");
    if (str.length === 0) {
      return;
    }
    if (str.length === 1) {
      this.key = tokens[0];
    }
    if (str[str.length - 1] === "" && str[str.length - 2] === "") {
      this.key = "+";
      tokens.pop();
      tokens.pop();
    } else {
      this.key = str[str.length - 1];
      tokens.pop();
    }
    this.key = this.key.toUpperCase();

    const mac = isMacPlatform();
    for (const token of tokens) {
      if (token === "Ctrl") {
        this.ctrlKey = true;
      } else if (token === "Shift") {
        this.shiftKey = true;
      } else if (token === "Alt") {
        this.altKey = true;
      } else if (token === "Meta") {
        this.metaKey = true;
      } else if (token === "CtrlCmd") {
        if (mac) {
          this.metaKey = true;
        } else {
          this.ctrlKey = true;
        }
      } else if (token === "WinCtrl") {
        if (mac) {
          this.ctrlKey = true;
        } else {
          this.metaKey = true;
        }
      } else {
        console.warn("Unknown modifier key:", token);
      }
    }

    // Convert to:
    // - A unique string representation for the shortcut, that can be used for
    //   debugging or comparisons.
    // - A human-readable representation (TODO: make locale-dependent,
    //   e.g., "Maj" instead of "Shift" in French)
    //
    // Note: on Mac, the human-readable representation of modifiers should use
    // symbols, and their order is different than on Windows.
    //
    if (this.ctrlKey) {
      this.str += "Ctrl+";
      this.prettyStr += mac ? "^ " : "Ctrl ";
    }
    if (this.altKey) {
      this.str += "Alt+";
      this.prettyStr += mac ? "⌥ " : "Alt ";
    }
    if (this.shiftKey) {
      this.str += "Shift+";
      this.prettyStr += mac ? "⇧ " : "Shift ";
    }
    if (this.metaKey) {
      this.str += "Meta+";
      this.prettyStr += mac ? "⌘ " : "Win ";
    }
    this.str += this.key;
    this.prettyStr += this.key;
  }

  /**
   * Returns whether this shortcut matches the given keyboard event. Note that
   * this means an exact match, for example if the user presses Ctrl+Shift+Z,
   * then `Shortcut("Ctrl+Z").matches(event)` is false.
   */
  matches(event: KeyboardEvent) {
    // TODO: Use event.code instead some cases? Example: "Shift+2" causes event.key = @
    return (
      event.key.toUpperCase() === this.key &&
      event.ctrlKey === this.ctrlKey &&
      event.altKey === this.altKey &&
      event.shiftKey === this.shiftKey &&
      event.metaKey === this.metaKey
    );
  }
}
