import type { Stage } from "@/lib/types";

export const STAGES: { id: Stage; label: string; hint: string }[] = [
  { id: "ingest", label: "Ingest", hint: "Crawl the website, read uploaded files and notes" },
  { id: "research", label: "Research", hint: "Web research when the evidence is sparse" },
  { id: "profile", label: "Profile", hint: "Extract product lines in catalog vocabulary" },
  { id: "retrieve", label: "Retrieve", hint: "Recall candidate FSC classes (BM25, embeddings, LLM)" },
  { id: "decide", label: "Decide", hint: "Score each candidate with the decision model" },
  { id: "rank", label: "Rank", hint: "Blend scores, apply thresholds, label confidence" },
];
