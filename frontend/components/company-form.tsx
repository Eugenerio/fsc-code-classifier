"use client";

import { LoaderCircle, Sparkles } from "lucide-react";

import { FileDropzone } from "@/components/file-dropzone";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { CompanyFormValues } from "@/lib/api";

interface CompanyFormProps {
  values: CompanyFormValues;
  onValuesChange: (patch: Partial<CompanyFormValues>) => void;
  files: File[];
  onFilesChange: (files: File[]) => void;
  running: boolean;
  onSubmit: () => void;
  onCancel: () => void;
  className?: string;
}

export function CompanyForm({
  values,
  onValuesChange,
  files,
  onFilesChange,
  running,
  onSubmit,
  onCancel,
  className,
}: CompanyFormProps) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>Company</CardTitle>
        <CardDescription>The name is required. Website, notes and files give the classifier more to read.</CardDescription>
      </CardHeader>
      <CardContent>
        <form
          className="flex flex-col gap-4"
          onSubmit={(e) => {
            e.preventDefault();
            onSubmit();
          }}
        >
          <fieldset disabled={running} className="m-0 flex min-w-0 flex-col gap-4 border-0 p-0">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="company_name">Company name *</Label>
              <Input
                id="company_name"
                required
                autoComplete="organization"
                placeholder="Acme Fasteners Inc"
                value={values.name}
                onChange={(e) => onValuesChange({ name: e.target.value })}
              />
            </div>

            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="website_url">Website</Label>
                <Input
                  id="website_url"
                  inputMode="url"
                  autoCapitalize="none"
                  placeholder="https://acme.com"
                  value={values.website_url}
                  onChange={(e) => onValuesChange({ website_url: e.target.value })}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="email_domain">Email domain</Label>
                <Input
                  id="email_domain"
                  autoCapitalize="none"
                  placeholder="acme.com"
                  value={values.email_domain}
                  onChange={(e) => onValuesChange({ email_domain: e.target.value })}
                />
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              <Label htmlFor="notes">Notes</Label>
              <Textarea
                id="notes"
                rows={4}
                placeholder="Products, services, certifications, anything that helps."
                value={values.notes}
                onChange={(e) => onValuesChange({ notes: e.target.value })}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <Label>Files</Label>
              <FileDropzone files={files} onChange={onFilesChange} />
            </div>
          </fieldset>

          <div className="flex gap-2">
            <Button type="submit" size="lg" className="flex-1" disabled={running || !values.name.trim()}>
              {running ? <LoaderCircle className="animate-spin" /> : ""}
              {running ? "Classifying…" : "Classify"}
            </Button>
            {running && (
              <Button type="button" size="lg" variant="outline" onClick={onCancel}>
                Cancel
              </Button>
            )}
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
