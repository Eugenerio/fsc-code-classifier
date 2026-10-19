import { Circle, CircleCheck, CircleMinus, CircleX, LoaderCircle } from "lucide-react";

import { Progress } from "@/components/ui/progress";
import { formatMs } from "@/lib/format";
import { STAGES } from "@/lib/stages";
import type { Stage } from "@/lib/types";
import type { StageState } from "@/lib/use-classifier";
import { cn } from "@/lib/utils";

function StageIcon({ status }: { status: StageState["status"] }) {
  switch (status) {
    case "running":
      return <LoaderCircle className="size-4 animate-spin text-primary" />;
    case "done":
      return <CircleCheck className="size-4 text-success" />;
    case "skipped":
      return <CircleMinus className="size-4 text-muted-foreground" />;
    case "error":
      return <CircleX className="size-4 text-destructive" />;
    default:
      return <Circle className="size-4 text-muted-foreground/40" />;
  }
}

interface PipelineTimelineProps {
  stages: Record<Stage, StageState>;
  /** One tile per stage instead of the detailed list; used once a result is shown. */
  compact?: boolean;
}

export function PipelineTimeline({ stages, compact }: PipelineTimelineProps) {
  const finished = STAGES.filter(({ id }) => ["done", "skipped", "error"].includes(stages[id].status)).length;

  if (compact) {
    return (
      <ol className="grid grid-cols-2 gap-2 sm:grid-cols-3" aria-label="Pipeline stages">
        {STAGES.map(({ id, label }) => {
          const stage = stages[id];
          return (
            <li
              key={id}
              title={stage.detail ?? undefined}
              className="flex items-center gap-2 rounded-lg border bg-muted/30 px-2.5 py-1.5 text-sm whitespace-nowrap"
            >
              <StageIcon status={stage.status} />
              <span className={cn("font-medium", stage.status === "skipped" && "text-muted-foreground")}>{label}</span>
              <span className="ml-auto font-mono text-xs text-muted-foreground tabular-nums">
                {stage.status === "skipped" ? "skipped" : stage.elapsedMs != null ? formatMs(stage.elapsedMs) : ""}
              </span>
            </li>
          );
        })}
      </ol>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <Progress value={(finished / STAGES.length) * 100} aria-label="Pipeline progress" />
      <ol className="flex flex-col" aria-label="Pipeline stages">
        {STAGES.map(({ id, label, hint }) => {
          const stage = stages[id];
          return (
            <li key={id} className="flex items-start gap-3 py-1.5">
              <div className="mt-0.5">
                <StageIcon status={stage.status} />
              </div>
              <div className="min-w-0 flex-1">
                <p className={cn("text-sm font-medium", stage.status === "pending" && "text-muted-foreground")}>
                  {label}
                  {stage.status === "skipped" && <span className="ml-2 text-xs font-normal text-muted-foreground">skipped</span>}
                </p>
                <p className={cn("text-xs text-muted-foreground", stage.status === "error" && "text-destructive")}>
                  {stage.detail ?? hint}
                </p>
              </div>
              <span className="font-mono text-xs text-muted-foreground tabular-nums">
                {stage.status !== "skipped" && stage.elapsedMs != null ? formatMs(stage.elapsedMs) : ""}
              </span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
