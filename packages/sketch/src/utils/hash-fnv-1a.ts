/* this code is taken from Claude - I should test it / revise this at some
 * point */

export function hashValue(
  value: string | number,
  seed: number = 2166136261,
): number {
  // Type checking - only accept strings or numbers
  if (typeof value !== "string" && typeof value !== "number") {
    throw new TypeError("Input must be a string or number");
  }

  // Handle number inputs consistently
  if (typeof value === "number") {
    // Handle special numeric values
    if (!Number.isFinite(value)) {
      return Number.isNaN(value) ? 0x01234567 : 0x89abcdef;
    }

    // For floating points, use a fixed precision to avoid issues like 0.1 + 0.2 !== 0.3
    if (Math.floor(value) !== value) {
      // Round to a reasonable precision (e.g., 10 decimal places)
      // This helps with floating point representation issues
      value = Number(value.toFixed(10));
    }

    // Convert number to a consistent string representation
    value = String(value);
  }

  const encoder = new TextEncoder();
  const bytes = Array.from(encoder.encode(value));

  // Standard FNV-1a hash implementation
  let hash = seed; // Use provided seed or FNV offset basis (32-bit)

  // Process each byte
  for (let i = 0; i < bytes.length; i++) {
    hash ^= bytes[i];
    hash = Math.imul(hash, 16777619); // FNV prime (32-bit)
  }

  return hash >>> 0; // Ensure unsigned 32-bit integer
}

export function combineHashes(
  valueHash: number,
  childrenHashes?: number[],
): number {
  // Start with the node's own value hash
  let result = valueHash & 0xffffffff;

  // Handle undefined or null childrenHashes
  if (!childrenHashes || childrenHashes.length === 0) {
    return result >>> 0;
  }

  // Add length information to make different sized arrays hash differently
  result = mixHash(result ^ (childrenHashes.length & 0xff));

  // Incorporate each child's hash with proper mixing and position awareness
  for (let i = 0; i < childrenHashes.length; i++) {
    const childHash = childrenHashes[i] || 0; // Default to 0 if child hash is undefined

    // Mix position information to ensure order matters
    // Each child contributes differently based on its position in the array
    const positionedChildHash = childHash ^ Math.imul(i + 1, 0x9e3779b9); // Prime constant

    // Mix in the child hash
    result ^= positionedChildHash;
    result = mixHash(result);
  }

  // Final avalanche mix for better distribution
  return finalizeHash(result);
}

// Helper function for hash mixing (based on MurmurHash3)
function mixHash(hash: number): number {
  hash = Math.imul(hash, 0x85ebca6b);
  // Correct bit rotation (right rotate by 13)
  hash = ((hash >>> 13) | (hash << 19)) >>> 0;
  hash = Math.imul(hash, 0xc2b2ae35);
  return hash >>> 0;
}

// Finalization mix - force all bits of the hash to avalanche
function finalizeHash(hash: number): number {
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b);
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35);
  hash ^= hash >>> 16;
  return hash >>> 0; // Ensure unsigned 32-bit integer
}
