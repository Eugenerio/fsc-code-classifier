"use client";

import { FileText, Image as ImageIcon, Upload, X } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { formatBytes } from "@/lib/format";
import { cn } from "@/lib/utils";

const EXTENSIONS = ["pdf", "png", "jpg", "jpeg", "webp", "txt", "md", "csv", "json", "html"];
const ACCEPT = EXTENSIONS.map((ext) => `.${ext}`).join(",");
// Mirrors the backend defaults (max_files / max_upload_mb); the backend stays the source of truth.
const MAX_FILES = 10;
const MAX_FILE_MB = 20;

const fileKey = (f: File) => `${f.name}:${f.size}:${f.lastModified}`;
const isImage = (f: File) => f.type.startsWith("image/");

interface FileDropzoneProps {
  files: File[];
  onChange: (files: File[]) => void;
}

/** Disabled state comes from an enclosing <fieldset disabled>. */
export function FileDropzone({ files, onChange }: FileDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function add(incoming: File[]) {
    const known = new Set(files.map(fileKey));
    const accepted: File[] = [];
    const rejected: string[] = [];
    for (const file of incoming) {
      const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
      if (known.has(fileKey(file))) continue;
      if (!EXTENSIONS.includes(ext)) rejected.push(`${file.name}: unsupported type`);
      else if (file.size > MAX_FILE_MB * 1024 * 1024) rejected.push(`${file.name}: over ${MAX_FILE_MB} MB`);
      else if (files.length + accepted.length >= MAX_FILES) rejected.push(`${file.name}: limit of ${MAX_FILES} files`);
      else accepted.push(file);
    }
    if (rejected.length) toast.error("Some files were skipped", { description: rejected.join("\n") });
    if (accepted.length) onChange([...files, ...accepted]);
  }

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          add(Array.from(e.dataTransfer.files));
        }}
        className={cn(
          "flex w-full flex-col items-center gap-1 rounded-lg border border-dashed px-3 py-4 text-center transition-colors outline-none",
          "hover:bg-muted/60 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50",
          dragging && "border-primary bg-muted",
        )}
      >
        <Upload className="size-4 text-muted-foreground" />
        <span className="text-sm font-medium">Drop files here or click to browse</span>
        <span className="text-xs text-muted-foreground">
          PDF, PNG/JPG/WEBP, TXT, MD, CSV · up to {MAX_FILES} files, {MAX_FILE_MB} MB each
        </span>
      </button>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPT}
        className="sr-only"
        tabIndex={-1}
        aria-label="Upload company files"
        onChange={(e) => {
          add(Array.from(e.target.files ?? []));
          e.target.value = ""; // lets the same file be picked again after removal
        }}
      />
      {files.length > 0 && (
        <ul className="flex flex-col gap-1">
          {files.map((file) => (
            <li key={fileKey(file)} className="flex items-center gap-2 rounded-md border bg-muted/30 py-1 pr-1 pl-2.5 text-sm">
              {isImage(file) ? (
                <ImageIcon className="size-4 shrink-0 text-muted-foreground" />
              ) : (
                <FileText className="size-4 shrink-0 text-muted-foreground" />
              )}
              <span className="min-w-0 flex-1 truncate" title={file.name}>
                {file.name}
              </span>
              <span className="shrink-0 font-mono text-xs text-muted-foreground">{formatBytes(file.size)}</span>
              <Button
                type="button"
                variant="ghost"
                size="icon-xs"
                aria-label={`Remove ${file.name}`}
                onClick={() => onChange(files.filter((f) => f !== file))}
              >
                <X />
              </Button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
