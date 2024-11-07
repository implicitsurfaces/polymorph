import xxhash from "xxhash-wasm";

const hasher = xxhash();

export class Hasher {
  private str: string;
  constructor() {
    this.str = "";
  }

  addString(str: string): Hasher {
    this.str += str;
    return this;
  }

  addNumber(num: number): Hasher {
    this.addString(num.toString());
    return this;
  }

  addBoolean(bool: boolean): Hasher {
    this.addString(bool.toString());
    return this;
  }

  addNull(): Hasher {
    this.addString("null");
    return this;
  }

  addUndefined(): Hasher {
    this.addString("undefined");
    return this;
  }

  addClass(obj: object): Hasher {
    this.addString(obj.constructor.name);
    return this;
  }

  async done(): Promise<string> {
    return (await hasher).h64ToString(this.str);
  }
}
