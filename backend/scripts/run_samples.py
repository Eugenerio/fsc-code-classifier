"""Run the three sample companies through the pipeline and print the FSC codes.

    cd backend && uv run python -m scripts.run_samples [--json] [--only NAME] [--pdf PATH]

`--pdf` attaches a capability-statement PDF to the Lone Star sample (the PDF is not in the repo).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
from pathlib import Path

from app.pipeline import run_pipeline, startup
from app.schemas import ClassifyResponse, CompanyInput, UploadedFile

SAMPLES = [
    CompanyInput(name="H & R PARTS CO INC", website_url="https://www.hrpartsco.com/"),
    CompanyInput(name="LOOS & CO INC", website_url="https://loosco.com/"),
    CompanyInput(
        name="Lone Star Downhole Product",
        # Stand-in for the capability-statement PDF, which is not in the repo (see --pdf).
        notes="Downhole tools and products for oil and gas drilling and well servicing.",
    ),
]


def load_pdf(path: Path) -> UploadedFile:
    return UploadedFile(
        filename=path.name,
        content_type=mimetypes.guess_type(path.name)[0] or "application/pdf",
        data=path.read_bytes(),
    )


def render(result: ClassifyResponse) -> str:
    lines = [f"\n=== {result.company.name} ({result.total_ms / 1000:.1f}s) ==="]
    lines.append(
        f"extractor={result.backend.extractor}  decider={result.backend.decider_model or result.backend.decider}  "
        f"recall={'+'.join(result.backend.retrieval) or '-'}"
    )
    lines += [f"  ! {w}" for w in result.warnings]
    lines.append("stages: " + "  ".join(f"{s.stage}:{s.elapsed_ms}ms" for s in result.stages if s.status != "skipped"))
    profile = result.profile
    lines.append(f"profile: {profile.summary or '(empty)'}")
    lines.append("  types: " + (", ".join(profile.business_types) or "-"))
    lines += [f"  - {p.name} [{p.role}]: {p.description}" for p in profile.products]
    lines.append("FSC codes:")
    for m in result.fsc_codes:
        lines.append(f"  {m.code}  {m.score:.2f} {m.confidence:<6} {m.description} ({m.group_name})")
        if m.rationale:
            lines.append(f"        {m.rationale}")
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print the raw ClassifyResponse list as JSON")
    parser.add_argument("--only", metavar="TEXT", help="run only samples whose name contains TEXT (case-insensitive)")
    parser.add_argument("--pdf", type=Path, help="capability-statement PDF for the Lone Star sample")
    args = parser.parse_args()

    samples = [s for s in SAMPLES if not args.only or args.only.lower() in s.name.lower()]
    if not samples:
        parser.error(f"no sample matches {args.only!r}: " + ", ".join(s.name for s in SAMPLES))
    pdf = [load_pdf(args.pdf)] if args.pdf else []

    await startup()  # build indexes once, then run the samples concurrently (also exercises concurrent safety)
    results = await asyncio.gather(*(run_pipeline(s, pdf if s.name.startswith("Lone Star") else []) for s in samples))
    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
    else:
        print("\n".join(render(r) for r in results))


if __name__ == "__main__":
    asyncio.run(main())
