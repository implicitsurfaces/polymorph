import { NodeId, AnyNodeData } from "./Node";

// This file contains helper functions to implement the `dataFromAny` static
// method of each node type. In particular, it requires error handling to
// validate the parsed input data, which is handled by the helper functions
// found here.

/**
 * This helper function checks that the given `property` exists in the
 * given `data` and that it is of type string.
 *
 * If it does, then this function returns the boolean property value.
 *
 * Otherwise, this function throws.
 */
export function asString(data: AnyNodeData, property: string): string {
  if (property in data) {
    const b = data[property];
    if (typeof b === "string") {
      return b;
    } else {
      throw Error(
        `Property "${property}" is not of type string in the given data (=${data}).`,
      );
    }
  } else {
    throw Error(
      `Missing string property "${property}" in the given data (=${data}).`,
    );
  }
}

/**
 * This helper function checks that the given `property` exists in the
 * given `data` and that it is of type number.
 *
 * If it does, then this function returns the number property value.
 *
 * Otherwise, this function throws.
 */
export function asNumber(data: AnyNodeData, property: string): number {
  if (property in data) {
    const b = data[property];
    if (typeof b === "number") {
      return b;
    } else {
      throw Error(
        `Property "${property}" is not of type number in the given data (=${data}).`,
      );
    }
  } else {
    throw Error(
      `Missing number property "${property}" in the given data (=${data}).`,
    );
  }
}

/**
 * This helper function checks that the given `property` exists in the
 * given `data` and that it is of type boolean.
 *
 * If it does, then this function returns the boolean property value.
 *
 * Otherwise, this function throws.
 */
export function asBoolean(data: AnyNodeData, property: string): boolean {
  if (property in data) {
    const b = data[property];
    if (typeof b === "boolean") {
      return b;
    } else {
      throw Error(
        `Property "${property}" is not of type boolean in the given data (=${data}).`,
      );
    }
  } else {
    throw Error(
      `Missing boolean property "${property}" in the given data (=${data}).`,
    );
  }
}

/**
 * This helper function checks that the given `property` exists in the
 * given `data` and that it is of type string.
 *
 * If it does, then this function returns the string as a `NodeId`.
 *
 * Otherwise, this function throws.
 */
export function asNodeId(data: AnyNodeData, property: string): NodeId {
  if (property in data) {
    const id = data[property];
    if (typeof id === "string") {
      return id;
    } else {
      throw Error(
        `ID property "${property}" is not of type string in the given data (=${data}).`,
      );
    }
  } else {
    throw Error(
      `Missing ID property "${property}" in the given data (=${data}).`,
    );
  }
}

/**
 * This helper function checks that the given `property` exists in the
 * given `data` and that it is of type string array.
 *
 * If it does, then this function returns the array as `NodeId[]`.
 *
 * Otherwise, this function throws.
 */
export function asNodeIdArray(data: AnyNodeData, property: string): NodeId[] {
  if (property in data) {
    const a = data[property];
    if (Array.isArray(a)) {
      for (const id of a) {
        if (typeof id != "string") {
          throw Error(
            `Found non-ID value in property "${property}" in the given data (=${data}).`,
          );
        }
      }
      return a;
    } else {
      throw Error(
        `Property "${property}" is not of type array of IDs in the given data (=${data}).`,
      );
    }
  } else {
    throw Error(
      `Missing array of IDs property "${property}" in the given data (=${data}).`,
    );
  }
}
