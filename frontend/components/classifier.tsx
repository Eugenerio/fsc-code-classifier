"use client";

import { useRef, useState } from "react";
import { toast } from "sonner";

import { BackendStatus } from "@/components/backend-status";
import { CompanyForm } from "@/components/company-form";
import { ResultsPanel } from "@/components/results-panel";
import { ThemeToggle } from "@/components/theme-toggle";
import { buildForm, type CompanyFormValues } from "@/lib/api";
import { useClassifier } from "@/lib/use-classifier";

const EMPTY: CompanyFormValues = { name: "", website_url: "", email_domain: "", notes: "" };

export function Classifier() {
  const [values, setValues] = useState(EMPTY);
  const [files, setFiles] = useState<File[]>([]);
  const { state, run, cancel } = useClassifier();
  const resultsRef = useRef<HTMLDivElement>(null);

  function submit() {
    if (!values.name.trim()) return;
    void run(buildForm(values, files));
    // Stacked layout: bring the live results into view.
    if (window.matchMedia("(max-width: 1023px)").matches) {
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function cancelRun() {
    cancel();
    toast("Classification cancelled");
  }

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 px-4 py-6 sm:px-6 lg:py-8">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight sm:text-2xl">FSC Code Classifier</h1>
          <p className="text-sm text-muted-foreground">
            Find the 4-digit Federal Supply Classification codes a company sells under.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <BackendStatus recheck={state.phase === "error"} />
          <ThemeToggle />
        </div>
      </header>

      <main className="grid gap-6 lg:grid-cols-[22rem_minmax(0,1fr)] lg:items-start xl:grid-cols-[24rem_minmax(0,1fr)]">
        <CompanyForm
          className="lg:sticky lg:top-6"
          values={values}
          onValuesChange={(patch) => setValues((v) => ({ ...v, ...patch }))}
          files={files}
          onFilesChange={setFiles}
          running={state.phase === "running"}
          onSubmit={submit}
          onCancel={cancelRun}
        />
        <div ref={resultsRef} className="min-w-0 scroll-mt-4">
          <ResultsPanel state={state} onRetry={submit} />
        </div>
      </main>
    </div>
  );
}
