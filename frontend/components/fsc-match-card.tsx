import { Badge } from "@/components/ui/badge";
import { Progress, ProgressLabel, ProgressValue } from "@/components/ui/progress";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Confidence, FscMatch } from "@/lib/types";
import { cn } from "@/lib/utils";

// Full class names (no string building) so Tailwind can see them.
const CONFIDENCE_STYLE: Record<Confidence, { badge: string; bar: string }> = {
  high: { badge: "bg-success/15 text-success", bar: "[&_[data-slot=progress-indicator]]:bg-success" },
  medium: { badge: "bg-warning/15 text-warning", bar: "[&_[data-slot=progress-indicator]]:bg-warning" },
  low: { badge: "bg-muted text-muted-foreground", bar: "[&_[data-slot=progress-indicator]]:bg-muted-foreground" },
};

const percent = (value: number) => `${Math.round(value * 100)}%`;

export function FscMatchCard({ match, rank }: { match: FscMatch; rank: number }) {
  const { decider_probability: probability } = match;
  const style = CONFIDENCE_STYLE[match.confidence];

  return (
    <li className="grid gap-x-5 gap-y-2 rounded-xl bg-card p-4 ring-1 ring-foreground/10 sm:grid-cols-[5.5rem_1fr]">
      <div className="flex items-baseline gap-2 sm:flex-col sm:gap-1">
        <span className="font-mono text-4xl leading-none font-semibold tracking-tight tabular-nums">{match.code}</span>
        <span className="text-xs text-muted-foreground">#{rank}</span>
      </div>

      <div className="flex min-w-0 flex-col gap-3">
        <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-2">
          <div className="min-w-0">
            <h3 className="leading-snug font-medium">{match.description}</h3>
            <p className="text-xs text-muted-foreground">
              Group {match.group} · {match.group_name}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-1.5">
            <Badge className={cn("capitalize", style.badge)}>{match.confidence}</Badge>
            <Tooltip>
              <TooltipTrigger render={<span />}>
                <Badge variant="outline" className="font-mono">
                  {match.sos_ric}
                </Badge>
              </TooltipTrigger>
              <TooltipContent>SOS RIC: source-of-supply routing code for this class</TooltipContent>
            </Tooltip>
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <Progress
            value={match.score * 100}
            aria-label={`Score for ${match.code}`}
            className={cn("gap-1.5 [&_[data-slot=progress-track]]:h-2", style.bar)}
          >
            <ProgressLabel className="text-xs font-normal text-muted-foreground">Score</ProgressLabel>
            <ProgressValue className="font-mono text-xs text-foreground" />
          </Progress>
          <p className="text-xs text-muted-foreground">
            {probability != null && <>Decider {percent(probability)} · </>}Retrieval {percent(match.retrieval_score)}
          </p>
        </div>

        {match.matched_products.length > 0 && (
          <ul className="flex flex-wrap gap-1.5" aria-label="Matched products">
            {match.matched_products.map((product) => (
              <li key={product}>
                <Badge variant="secondary">{product}</Badge>
              </li>
            ))}
          </ul>
        )}

        {match.rationale && (
          <p className="border-l-2 pl-3 text-sm text-muted-foreground">{match.rationale}</p>
        )}
      </div>
    </li>
  );
}
