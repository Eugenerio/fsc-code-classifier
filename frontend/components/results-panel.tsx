"use client";

import { Ban, Inbox, RotateCw, ServerOff, TriangleAlert } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { CopyButton } from "@/components/copy-button";
import { FscCodesTab } from "@/components/fsc-codes-tab";
import { PipelineTimeline } from "@/components/pipeline-timeline";
import { ProfileTab } from "@/components/profile-tab";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { API_URL } from "@/lib/api";
import { formatMs } from "@/lib/format";
import type { ClassifyResponse } from "@/lib/types";
import type { ClassifierState } from "@/lib/use-classifier";

function Stopwatch() {
  const [ms, setMs] = useState(0);
  useEffect(() => {
    const start = performance.now();
    const id = setInterval(() => setMs(performance.now() - start), 100);
    return () => clearInterval(id);
  }, []);
  return <span className="font-mono text-xs text-muted-foreground tabular-nums">{formatMs(ms)}</span>;
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-2 py-16 text-center">
      <Inbox className="size-8 text-muted-foreground/60" />
      <p className="font-medium">No classification yet</p>
      <p className="max-w-sm text-sm text-muted-foreground">
        Enter a company on the left, then press Classify. The 4-digit FSC codes appear here.
      </p>
    </div>
  );
}

function ResultSkeleton() {
  return (
    <div className="flex flex-col gap-3" aria-hidden>
      {[0, 1, 2].map((i) => (
        <div key={i} className="flex gap-4 rounded-xl p-4 ring-1 ring-foreground/10">
          <Skeleton className="h-10 w-20" />
          <div className="flex flex-1 flex-col gap-2">
            <Skeleton className="h-4 w-2/3" />
            <Skeleton className="h-2 w-full" />
            <Skeleton className="h-3 w-1/2" />
          </div>
        </div>
      ))}
    </div>
  );
}

function BackendUnreachable({ message }: { message: string }) {
  return (
    <>
      <Alert variant="destructive">
        <ServerOff />
        <AlertTitle>Backend unreachable</AlertTitle>
        <AlertDescription>{message}.</AlertDescription>
      </Alert>
      <div className="flex flex-col gap-2 rounded-lg border p-3 text-sm">
        <p className="font-medium">Start the API in another terminal</p>
        <pre className="overflow-x-auto rounded-md bg-muted p-2 font-mono text-xs">
          {"cd backend\nuv run uvicorn app.main:app --port 8000"}
        </pre>
        <p className="text-muted-foreground">
          No backend handy? Run <code className="font-mono text-foreground">pnpm mock</code> in{" "}
          <code className="font-mono text-foreground">frontend/</code> for a stub API. The address comes from{" "}
          <code className="font-mono text-foreground">NEXT_PUBLIC_API_URL</code> (currently {API_URL}).
        </p>
      </div>
    </>
  );
}

function ResultTabs({ data }: { data: ClassifyResponse }) {
  const json = useMemo(() => JSON.stringify(data, null, 2), [data]);
  return (
    <Tabs defaultValue="codes">
      <TabsList>
        <TabsTrigger value="codes">FSC codes ({data.fsc_codes.length})</TabsTrigger>
        <TabsTrigger value="profile">Profile</TabsTrigger>
        <TabsTrigger value="json">Raw JSON</TabsTrigger>
      </TabsList>
      <TabsContent value="codes" className="pt-2">
        <FscCodesTab matches={data.fsc_codes} />
      </TabsContent>
      <TabsContent value="profile" className="pt-2">
        <ProfileTab data={data} />
      </TabsContent>
      <TabsContent value="json" className="flex flex-col gap-2 pt-2">
        <div className="flex justify-end">
          <CopyButton text={json} label="Copy JSON" successMessage="Copied JSON" variant="outline" />
        </div>
        <ScrollArea className="h-[30rem] rounded-lg border bg-muted/30">
          <pre className="p-3 font-mono text-xs break-words whitespace-pre-wrap">{json}</pre>
        </ScrollArea>
      </TabsContent>
    </Tabs>
  );
}

function BackendFooter({ data }: { data: ClassifyResponse }) {
  const { backend } = data;
  const decider = backend.decider_model ? `${backend.decider} (${backend.decider_model})` : backend.decider;
  return (
    <CardFooter className="flex-wrap gap-x-4 gap-y-1 font-mono text-xs text-muted-foreground">
      <span>extractor: {backend.extractor}</span>
      <span>decider: {decider}</span>
      <span>retrieval: {backend.retrieval.join(" + ") || "none"}</span>
      <span>total: {formatMs(data.total_ms)}</span>
    </CardFooter>
  );
}

interface ResultsPanelProps {
  state: ClassifierState;
  onRetry: () => void;
  className?: string;
}

export function ResultsPanel({ state, onRetry, className }: ResultsPanelProps) {
  const { phase, stages, result, error } = state;
  const started = phase !== "idle";

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Results</CardTitle>
        <CardDescription>
          {phase === "running" ? "Reading the evidence and matching FSC classes…" : "Ranked FSC classes for the company."}
        </CardDescription>
        {phase === "running" && (
          <CardAction>
            <Stopwatch />
          </CardAction>
        )}
      </CardHeader>

      <CardContent className="flex flex-col gap-4" aria-live="polite">
        {phase === "idle" && <EmptyState />}

        {phase === "error" && error && (
          <>
            {error.unreachable ? (
              <BackendUnreachable message={error.message} />
            ) : (
              <Alert variant="destructive">
                <TriangleAlert />
                <AlertTitle>Classification failed</AlertTitle>
                <AlertDescription>{error.message}</AlertDescription>
              </Alert>
            )}
            <div>
              <Button variant="outline" onClick={onRetry}>
                <RotateCw /> Try again
              </Button>
            </div>
          </>
        )}

        {phase === "cancelled" && (
          <Alert>
            <Ban />
            <AlertTitle>Cancelled</AlertTitle>
            <AlertDescription>The run was stopped before a result arrived. Press Classify to start again.</AlertDescription>
          </Alert>
        )}

        {started && !(phase === "error" && error?.unreachable) && (
          <PipelineTimeline stages={stages} compact={phase === "done"} />
        )}

        {phase === "running" && <ResultSkeleton />}

        {phase === "done" && result && (
          <>
            {result.warnings.length > 0 && (
              <Alert>
                <TriangleAlert />
                <AlertTitle>{result.warnings.length === 1 ? "Warning" : `${result.warnings.length} warnings`}</AlertTitle>
                <AlertDescription>
                  <ul className="list-disc pl-4">
                    {result.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </AlertDescription>
              </Alert>
            )}
            <ResultTabs data={result} />
          </>
        )}
      </CardContent>

      {phase === "done" && result && <BackendFooter data={result} />}
    </Card>
  );
}
