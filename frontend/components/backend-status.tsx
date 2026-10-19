"use client";

import { useEffect, useState } from "react";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { API_URL, getHealth } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { HealthResponse } from "@/lib/types";

type Status = { kind: "checking" } | { kind: "offline" } | { kind: "online"; health: HealthResponse };

/** Pings GET /api/health on mount and whenever `recheck` changes. */
export function BackendStatus({ recheck }: { recheck: unknown }) {
  const [status, setStatus] = useState<Status>({ kind: "checking" });

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal)
      .then((health) => setStatus({ kind: "online", health }))
      .catch(() => !controller.signal.aborted && setStatus({ kind: "offline" }));
    return () => controller.abort();
  }, [recheck]);

  const online = status.kind === "online";
  const text =
    status.kind === "checking"
      ? "Checking backend"
      : online
        ? `Backend online · ${status.health.decider} decider`
        : "Backend offline";
  const detail = online
    ? `Gemini ${status.health.gemini ? "on" : "off"} · Jev ${status.health.jev ? "on" : "off"} · ${status.health.fsc_count} FSC classes · retrieval: ${status.health.retrieval.join(" + ")}`
    : `${API_URL}`;

  return (
    <Tooltip>
      <TooltipTrigger
        render={<span />}
        className="inline-flex h-7 items-center gap-2 rounded-full border px-2.5 text-xs text-muted-foreground"
      >
        <span
          className={cn(
            "size-1.5 rounded-full",
            status.kind === "checking" && "bg-muted-foreground/50",
            status.kind === "offline" && "bg-destructive",
            online && "bg-success",
          )}
        />
        {text}
      </TooltipTrigger>
      <TooltipContent>{detail}</TooltipContent>
    </Tooltip>
  );
}
