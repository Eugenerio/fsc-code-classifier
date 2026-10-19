import { CopyButton } from "@/components/copy-button";
import { FscMatchCard } from "@/components/fsc-match-card";
import type { FscMatch } from "@/lib/types";

export function FscCodesTab({ matches }: { matches: FscMatch[] }) {
  if (matches.length === 0) {
    return (
      <p className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
        No FSC class matched this company. Add a website, notes or a catalog file and try again.
      </p>
    );
  }

  const codes = matches.map((m) => m.code).join(", ");

  return (
    <div className="flex flex-col gap-4">
      <section
        aria-label="Recommended FSC codes"
        className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-muted/50 p-4"
      >
        <div className="min-w-0">
          <p className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
            {matches.length} recommended FSC {matches.length === 1 ? "code" : "codes"}
          </p>
          <p className="font-mono text-2xl font-semibold tracking-tight break-words" data-testid="codes-summary">
            {codes}
          </p>
        </div>
        <CopyButton text={codes} label="Copy codes" successMessage={`Copied ${matches.length} codes`} />
      </section>
      <ol className="flex flex-col gap-3">
        {matches.map((match, i) => (
          <FscMatchCard key={match.code} match={match} rank={i + 1} />
        ))}
      </ol>
    </div>
  );
}
