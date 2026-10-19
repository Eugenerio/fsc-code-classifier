// Run with `pnpm test`. Imports the TS source directly (Node strips the types).
import assert from "node:assert/strict";
import { test } from "node:test";

import { NdjsonSplitter } from "../lib/ndjson.ts";

const bytes = (s) => new TextEncoder().encode(s);

test("joins a line split across chunks", () => {
  const splitter = new NdjsonSplitter();
  assert.deepEqual(splitter.push(bytes('{"a":1}\n{"b"')), ['{"a":1}']);
  assert.deepEqual(splitter.push(bytes(':2}\n')), ['{"b":2}']);
  assert.deepEqual(splitter.flush(), []);
});

test("returns several lines from one chunk and skips blanks", () => {
  const splitter = new NdjsonSplitter();
  assert.deepEqual(splitter.push(bytes('{"a":1}\n\n{"b":2}\n')), ['{"a":1}', '{"b":2}']);
});

test("flush returns a final line without newline", () => {
  const splitter = new NdjsonSplitter();
  assert.deepEqual(splitter.push(bytes('{"a":1}')), []);
  assert.deepEqual(splitter.flush(), ['{"a":1}']);
});

test("keeps multi-byte characters intact when a chunk cuts them in half", () => {
  const splitter = new NdjsonSplitter();
  const all = bytes('{"name":"Müller € ✓"}\n');
  const out = [];
  for (const byte of all) out.push(...splitter.push(Uint8Array.of(byte)));
  assert.deepEqual(out, ['{"name":"Müller € ✓"}']);
});
