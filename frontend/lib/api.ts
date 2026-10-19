import { NdjsonSplitter } from "@/lib/ndjson";
import type { HealthResponse, StreamEvent } from "@/lib/types";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

/** The backend answered, but with an error (HTTP status or an `error` stream event). */
export class ApiError extends Error {}

/** The backend could not be reached at all (not running, wrong URL, CORS). */
export class UnreachableError extends Error {}

export interface CompanyFormValues {
  name: string;
  website_url: string;
  email_domain: string;
  notes: string;
}

export function buildForm(values: CompanyFormValues, files: File[]): FormData {
  const form = new FormData();
  form.set("company_name", values.name.trim());
  for (const key of ["website_url", "email_domain", "notes"] as const) {
    const value = values[key].trim();
    if (value) form.set(key, value);
  }
  for (const file of files) form.append("files", file);
  return form;
}

async function errorMessage(res: Response): Promise<string> {
  const fallback = `Backend returned HTTP ${res.status}`;
  try {
    const body = await res.json();
    const detail = body?.detail ?? body?.message;
    if (typeof detail === "string") return detail;
    // FastAPI validation errors: [{loc, msg, type}, ...]
    if (Array.isArray(detail)) return detail.map((d) => d?.msg ?? JSON.stringify(d)).join("; ");
  } catch {
    // body was not JSON
  }
  return fallback;
}

async function request(path: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(`${API_URL}${path}`, init);
  } catch (err) {
    if (init.signal?.aborted) throw err;
    throw new UnreachableError(`Could not reach the backend at ${API_URL}`);
  }
}

export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const res = await request("/api/health", { signal });
  if (!res.ok) throw new ApiError(await errorMessage(res));
  return res.json();
}

/** POSTs the form and calls `onEvent` for every NDJSON line as it arrives. Resolves when the stream ends. */
export async function classifyStream(
  form: FormData,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await request("/api/classify/stream", { method: "POST", body: form, signal });
  if (!res.ok) throw new ApiError(await errorMessage(res));
  if (!res.body) throw new ApiError("The backend returned an empty response");

  const splitter = new NdjsonSplitter();
  const emit = (line: string) => {
    let event: StreamEvent;
    try {
      event = JSON.parse(line);
    } catch {
      throw new ApiError(`Malformed stream line: ${line.slice(0, 80)}`);
    }
    onEvent(event);
  };

  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    splitter.push(value).forEach(emit);
  }
  splitter.flush().forEach(emit);
}
