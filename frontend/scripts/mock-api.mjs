#!/usr/bin/env node
// Dependency-free stand-in for the FastAPI backend (GET /api/health, POST /api/classify/stream).
// Usage: node scripts/mock-api.mjs   (PORT=8000 by default)
// Magic company names: containing "fail" -> ErrorEvent, "slow" -> 3x slower, "empty" -> no codes.
import { createServer } from "node:http";

const PORT = Number(process.env.PORT ?? 8000);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const STAGES = [
  ["ingest", 700, "3 website pages, 1 file, notes"],
  ["research", 0, "Enough evidence, web research skipped"],
  ["profile", 900, "6 product lines (heuristic extractor)"],
  ["retrieve", 800, "42 candidates from bm25"],
  ["decide", 1000, "42 candidates scored by retrieval"],
  ["rank", 150, "6 codes kept"],
];

const MATCHES = [
  ["5305", "Screws", "S9C/AX", "53", "Hardware and Abrasives", 0.91, "high", 0.94, 0.82, ["hex cap screws", "machine screws"], "Sells socket and hex cap screws in stainless and alloy steel."],
  ["5306", "Bolts", "S9C/AX", "53", "Hardware and Abrasives", 0.88, "high", 0.9, 0.78, ["hex cap screws"], "Carriage, flange and hex bolts are listed on the product pages."],
  ["5310", "Nuts and Washers", "S9C/AX", "53", "Hardware and Abrasives", 0.8, "high", 0.83, 0.7, ["nuts and washers"], "Distributes lock nuts, flat washers and lock washers."],
  ["5315", "Nails, Keys, and Pins", "S9C/AX", "53", "Hardware and Abrasives", 0.66, "medium", 0.68, 0.6, ["cotter pins", "dowel pins"], "Cotter pins and dowel pins appear in the fastener catalog."],
  ["5340", "Hardware, Commercial", "S9C/AX", "53", "Hardware and Abrasives", 0.58, "medium", null, 0.58, ["miscellaneous hardware"], null],
  ["4730", "Hose, Pipe, Tube, Lubrication, and Railing Fittings", "S9P/AX", "47", "Pipe, Tubing, Hose, and Fittings", 0.47, "low", null, 0.47, ["pipe fittings"], "Matched on 'fittings' keyword only; weak evidence."],
];

function sources(fileNames) {
  return [
    { kind: "notes", label: "notes", chars: 240, ok: true, error: null, meta: {} },
    { kind: "website", label: "https://example.com/", chars: 5120, ok: true, error: null, meta: {} },
    { kind: "website", label: "https://example.com/products", chars: 12840, ok: true, error: null, meta: {} },
    { kind: "website", label: "https://example.com/about", chars: 0, ok: false, error: "HTTP 404", meta: {} },
    ...fileNames.map((label) => ({
      kind: "file",
      label,
      chars: 18400,
      ok: true,
      error: null,
      meta: { ocr: label.endsWith(".pdf"), pages: 7 },
    })),
  ];
}

function buildResult(form, stageEvents) {
  const name = form.fields.company_name;
  const none = /empty/i.test(name);
  return {
    company: {
      name,
      website_url: form.fields.website_url ?? null,
      email_domain: form.fields.email_domain ?? null,
      notes: form.fields.notes ?? null,
    },
    profile: {
      name,
      summary: `${name} supplies fasteners and commercial hardware to industrial and defense customers.`,
      business_types: ["distributor", "fastener supplier"],
      products: [
        { name: "hex cap screws", description: "Stainless and alloy steel hex head cap screws.", keywords: ["bolts", "hex head", "cap screw"], role: "distributor" },
        { name: "nuts and washers", description: "Lock nuts, flat and lock washers.", keywords: ["lock nut", "flat washer"], role: "distributor" },
        { name: "cotter pins", description: "Cotter and dowel pins.", keywords: ["split pin", "dowel"], role: "reseller" },
        { name: "pipe fittings", description: "", keywords: ["elbow", "coupling"], role: "unknown" },
      ],
      keywords: ["fasteners", "hardware", "stainless steel", "mil-spec"],
      industries: ["defense", "aerospace", "construction"],
    },
    fsc_codes: none
      ? []
      : MATCHES.map(([code, description, sos_ric, group, group_name, score, confidence, probability, retrieval, products, rationale]) => ({
          code, description, sos_ric, group, group_name, score, confidence,
          decider_probability: probability,
          retrieval_score: retrieval,
          matched_products: products,
          rationale,
        })),
    sources: sources(form.files),
    warnings: ["GEMINI_API_KEY is not set: using the heuristic profile extractor.", "TYPESAFE_API_KEY is not set: ranking uses retrieval scores only."],
    stages: stageEvents,
    backend: { extractor: "heuristic", decider: "retrieval", decider_model: null, retrieval: ["bm25"] },
    total_ms: stageEvents.reduce((sum, s) => sum + s.elapsed_ms, 0),
  };
}

/** Minimal multipart/form-data reader: text fields plus uploaded file names. */
function parseMultipart(body, contentType) {
  const boundary = /boundary=(?:"([^"]+)"|([^;]+))/.exec(contentType ?? "");
  const form = { fields: {}, files: [] };
  if (!boundary) return form;
  for (const part of body.toString("latin1").split(`--${boundary[1] ?? boundary[2]}`)) {
    const head = /name="([^"]+)"(?:; filename="([^"]*)")?/.exec(part.split("\r\n\r\n")[0]);
    if (!head) continue;
    const value = part.slice(part.indexOf("\r\n\r\n") + 4).replace(/\r\n$/, "");
    if (head[2]) form.files.push(head[2]);
    else form.fields[head[1]] = Buffer.from(value, "latin1").toString("utf8");
  }
  return form;
}

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks);
}

function json(res, status, payload) {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(payload));
}

async function classifyStream(req, res) {
  const form = parseMultipart(await readBody(req), req.headers["content-type"]);
  const name = form.fields.company_name?.trim();
  if (!name) return json(res, 422, { detail: "company_name is required" });

  res.writeHead(200, { "content-type": "application/x-ndjson", "x-accel-buffering": "no", "cache-control": "no-cache" });
  let closed = false;
  res.on("close", () => (closed = true));
  const send = (event) => res.write(`${JSON.stringify(event)}\n`);
  const pace = /slow/i.test(name) ? 3 : 1;

  const timings = [];
  for (const [stage, ms, detail] of STAGES) {
    if (closed) return;
    send({ event: "stage", stage, status: "running", detail: null, elapsed_ms: null });
    await sleep(ms * pace);
    if (/fail/i.test(name) && stage === "profile") {
      send({ event: "stage", stage, status: "error", detail: "Profile extraction crashed", elapsed_ms: ms });
      send({ event: "error", message: "Mock failure requested by the company name." });
      return res.end();
    }
    const status = ms === 0 ? "skipped" : "done";
    timings.push({ stage, status, elapsed_ms: ms, detail });
    send({ event: "stage", stage, status, detail, elapsed_ms: ms });
  }

  // Split the final line across two writes to exercise the client's chunk-boundary handling.
  const line = `${JSON.stringify({ event: "result", data: buildResult(form, timings) })}\n`;
  const cut = Math.floor(line.length / 2);
  res.write(line.slice(0, cut));
  await sleep(60);
  res.end(line.slice(cut));
}

createServer(async (req, res) => {
  res.setHeader("access-control-allow-origin", req.headers.origin ?? "*");
  res.setHeader("access-control-allow-headers", "*");
  res.setHeader("access-control-allow-methods", "GET, POST, OPTIONS");
  const { pathname } = new URL(req.url ?? "/", `http://${req.headers.host}`);

  if (req.method === "OPTIONS") return res.writeHead(204).end();
  if (req.method === "GET" && pathname === "/api/health") {
    return json(res, 200, { status: "ok", gemini: false, jev: false, decider: "retrieval", fsc_count: 580, retrieval: ["bm25"] });
  }
  if (req.method === "POST" && pathname === "/api/classify/stream") return classifyStream(req, res);
  json(res, 404, { detail: "Not found" });
}).listen(PORT, () => console.log(`mock API listening on http://localhost:${PORT}`));
