/**
 * Type of any `NodeData` property's value. This is essentially the same as a
 * parsed JSON value.
 */
export type AnyNodeDataValue =
  | string
  | number
  | boolean
  | null // [1]
  | readonly AnyNodeDataValue[]
  | AnyNodeData;

// [1] Unfortunately, we need to use `null` and not `undefined` for
// compatiblity with parsed JSON.
//
// This makes optional chaining uglier, for example `id = node?.id ?? null`
// instead of just `id = node?.id`, but it seems like a necessary evil.

/**
 * Type of any `NodeData`. This is essentially the same as a parsed JSON
 * object.
 */
export type AnyNodeData = { readonly [property: string]: AnyNodeDataValue };

// Note: we want node data to be immutable, and therefore arrays SHOULD be
// typed as `readonly T[]`:
//
// ```
// export interface CustomNodeData extends NodeData {
//   readonly nodeIds: readonly NodeId[];
// }
// ```
//
// This is why we need `readonly` in the `AnyNodeData` and `AnyNodeDataValue`
// type definitions, otherwise the snippet above would produce an error, because
// `readonly T[]` cannot be assigned to type `T[]`.
//
// Unfortunately, the following currently does not produce an error, but you
// SHOULD NOT do it:
//
// ```
// export interface CustomNodeData extends NodeData {
//   readonly nodeIds: NodeId[];
// }
// ```
//
// If the future we find a way to make the above an error, we will.
//
// The reason it does not currently produce an error is that `T[]` can be
// assigned to type `readonly T[]`, and therefore the type `{ foo: T
// [] }` does extend `{ foo: readonly T[]; }`.
