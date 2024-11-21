const FILLED_CHAR = "█";
const EMPTY_CHAR = " ";

type BooleanImageData = boolean[][];

export function booleansToASCII(
  imageData: BooleanImageData,
  double = false,
): string {
  const fillChar = double ? FILLED_CHAR + FILLED_CHAR : FILLED_CHAR;
  const emptyChar = double ? EMPTY_CHAR + EMPTY_CHAR : EMPTY_CHAR;

  return imageData
    .map((row) => row.map((pixel) => (pixel ? fillChar : emptyChar)).join(""))
    .join("\n");
}

export function intArrayToImageData(imageData: Uint8Array): BooleanImageData {
  const rowLength = Math.sqrt(imageData.length);
  const result: BooleanImageData = [];
  for (let i = 0; i < imageData.length; i += rowLength) {
    const row = [...imageData.slice(i, i + rowLength)].map(
      (value) => value > 0,
    );
    result.push(row);
  }
  return result;
}
