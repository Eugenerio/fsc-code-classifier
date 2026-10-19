/** Mirrors backend/app/schemas.py (response + stream types). Keep field names identical. */

export type SourceKind = "website" | "file" | "notes" | "web_research";
export type Role = "manufacturer" | "distributor" | "reseller" | "service" | "unknown";
export type CandidateSource = "bm25" | "embedding" | "llm";
export type Confidence = "high" | "medium" | "low";
export type Stage = "ingest" | "research" | "profile" | "retrieve" | "decide" | "rank";

export interface CompanyInput {
  name: string;
  website_url: string | null;
  email_domain: string | null;
  notes: string | null;
}

export interface SourceInfo {
  kind: SourceKind;
  label: string;
  chars: number;
  ok: boolean;
  error: string | null;
  meta: Record<string, unknown>;
}

export interface ProductLine {
  name: string;
  description: string;
  keywords: string[];
  role: Role;
}

export interface CompanyProfile {
  name: string;
  summary: string;
  business_types: string[];
  products: ProductLine[];
  keywords: string[];
  industries: string[];
}

export interface FscMatch {
  code: string;
  description: string;
  sos_ric: string;
  group: string;
  group_name: string;
  score: number;
  confidence: Confidence;
  decider_probability: number | null;
  retrieval_score: number;
  matched_products: string[];
  rationale: string | null;
}

export interface StageTiming {
  stage: Stage;
  status: "done" | "skipped" | "error";
  elapsed_ms: number;
  detail: string | null;
}

export interface BackendInfo {
  extractor: string;
  decider: "jev" | "gemini" | "retrieval";
  decider_model: string | null;
  retrieval: CandidateSource[];
}

export interface ClassifyResponse {
  company: CompanyInput;
  profile: CompanyProfile;
  fsc_codes: FscMatch[];
  sources: SourceInfo[];
  warnings: string[];
  stages: StageTiming[];
  backend: BackendInfo;
  total_ms: number;
}

// --------------------------------------------------------------------------- streaming (NDJSON)

export interface StageEvent {
  event: "stage";
  stage: Stage;
  status: "running" | "done" | "skipped" | "error";
  detail: string | null;
  elapsed_ms: number | null;
}

export interface ResultEvent {
  event: "result";
  data: ClassifyResponse;
}

export interface ErrorEvent {
  event: "error";
  message: string;
}

export type StreamEvent = StageEvent | ResultEvent | ErrorEvent;

// --------------------------------------------------------------------------- GET /api/health

export interface HealthResponse {
  status: "ok";
  gemini: boolean;
  jev: boolean;
  decider: BackendInfo["decider"];
  fsc_count: number;
  retrieval: CandidateSource[];
}
