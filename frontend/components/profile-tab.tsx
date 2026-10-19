import { Check } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import type { ClassifyResponse, SourceInfo } from "@/lib/types";

function BadgeList({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <section className="flex flex-col gap-2">
      <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">{title}</h3>
      <ul className="flex flex-wrap gap-1.5">
        {items.map((item) => (
          <li key={item}>
            <Badge variant="secondary">{item}</Badge>
          </li>
        ))}
      </ul>
    </section>
  );
}

function sourceNote(source: SourceInfo): string | null {
  const { ocr, pages } = source.meta;
  const parts = [ocr ? "OCR" : null, typeof pages === "number" ? `${pages} ${pages === 1 ? "page" : "pages"}` : null];
  return parts.filter(Boolean).join(" · ") || null;
}

function SourceLabel({ source }: { source: SourceInfo }) {
  if (source.kind === "website" && /^https?:\/\//.test(source.label)) {
    return (
      <a href={source.label} target="_blank" rel="noreferrer noopener" className="underline-offset-2 hover:underline">
        {source.label}
      </a>
    );
  }
  return <>{source.label}</>;
}

export function ProfileTab({ data }: { data: ClassifyResponse }) {
  const { profile, sources } = data;

  return (
    <div className="flex flex-col gap-5">
      <section className="flex flex-col gap-2">
        <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">Summary</h3>
        <p className="text-sm leading-relaxed">{profile.summary || "No summary could be built from the evidence."}</p>
      </section>

      <BadgeList title="Business types" items={profile.business_types} />

      <section className="flex flex-col gap-2">
        <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
          Product lines ({profile.products.length})
        </h3>
        {profile.products.length === 0 ? (
          <p className="text-sm text-muted-foreground">No product lines were extracted.</p>
        ) : (
          <ul className="flex flex-col gap-2">
            {profile.products.map((product) => (
              <li key={product.name} className="flex flex-col gap-1.5 rounded-lg border p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{product.name}</span>
                  <Badge variant={product.role === "unknown" ? "outline" : "secondary"} className="capitalize">
                    {product.role}
                  </Badge>
                </div>
                {product.description && <p className="text-sm text-muted-foreground">{product.description}</p>}
                {product.keywords.length > 0 && (
                  <p className="font-mono text-xs text-muted-foreground">{product.keywords.join(" · ")}</p>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      <BadgeList title="Keywords" items={profile.keywords} />
      <BadgeList title="Industries" items={profile.industries} />

      <Separator />

      <section className="flex flex-col gap-2">
        <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
          Sources ({sources.length})
        </h3>
        {sources.length === 0 ? (
          <p className="text-sm text-muted-foreground">No evidence was collected.</p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Source</TableHead>
                <TableHead className="text-right">Chars</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sources.map((source, i) => (
                <TableRow key={`${source.kind}:${source.label}:${i}`}>
                  <TableCell className="max-w-[18rem]">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="shrink-0">
                        {source.kind.replace("_", " ")}
                      </Badge>
                      <span className="truncate" title={source.label}>
                        <SourceLabel source={source} />
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{source.chars.toLocaleString("en-US")}</TableCell>
                  <TableCell>
                    {source.ok ? (
                      <span className="inline-flex items-center gap-1 text-success">
                        <Check className="size-3.5" /> OK
                        {sourceNote(source) && <span className="text-muted-foreground">· {sourceNote(source)}</span>}
                      </span>
                    ) : (
                      <span className="text-destructive">{source.error ?? "Failed"}</span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </section>
    </div>
  );
}
