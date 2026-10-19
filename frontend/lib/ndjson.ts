/** Splits a byte stream into NDJSON lines; network chunks may cut a line (or a UTF-8 character) anywhere. */
export class NdjsonSplitter {
  private decoder = new TextDecoder();
  private tail = "";

  push(chunk: Uint8Array): string[] {
    const lines = (this.tail + this.decoder.decode(chunk, { stream: true })).split("\n");
    this.tail = lines.pop() ?? "";
    return lines.filter((line) => line.trim());
  }

  /** Call once the stream ends: a final line may arrive without a trailing newline. */
  flush(): string[] {
    const rest = (this.tail + this.decoder.decode()).trim();
    this.tail = "";
    return rest ? [rest] : [];
  }
}
