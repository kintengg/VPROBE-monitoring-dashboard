"use client"

import Link from "next/link"
import { FileVideo, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"

interface QueueUploadZoneProps {
  activeCount: number
  queuedCount: number
  completedCount: number
  maxConcurrentUploads: number
}

export function QueueUploadZone({ activeCount, queuedCount, completedCount, maxConcurrentUploads }: QueueUploadZoneProps) {
  const uploadHref = "/?openAddVideo=1"

  return (
    <section className="relative overflow-hidden rounded-3xl border-2 border-dashed border-border bg-card p-8 shadow-elevated-sm transition-all hover:border-primary/50 hover:bg-secondary/30">
      <div className="relative z-10 flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-col items-start gap-4 text-left">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-secondary">
            <Upload className="h-8 w-8 text-muted-foreground" />
          </div>

          <div>
            <p className="text-white">
              <Link href={uploadHref} className="font-semibold text-primary transition-colors hover:text-primary/80 hover:underline">
                Open Surveillance
              </Link>
              {" "}to upload your video and track it here live, or{" "}
              <Link href={uploadHref} className="font-semibold text-primary transition-colors hover:text-primary/80 hover:underline">
                click here
              </Link>
              .
            </p>
            <p className="mt-2 text-sm text-muted-foreground">
              Background uploads continue here while you browse. Up to {maxConcurrentUploads} videos can run at the same time.
            </p>
          </div>
        </div>

        <div className="flex flex-col items-start gap-4 lg:items-end">
          <Button asChild className="rounded-2xl px-5">
            <Link href={uploadHref}>
              Open Surveillance
            </Link>
          </Button>

          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="min-w-[92px] rounded-2xl border border-border bg-secondary/60 px-3 py-2">
              <p className="text-[11px] text-muted-foreground">Active</p>
              <p className="text-xl font-semibold text-white">{activeCount}</p>
            </div>
            <div className="min-w-[92px] rounded-2xl border border-border bg-secondary/60 px-3 py-2">
              <p className="text-[11px] text-muted-foreground">Queued</p>
              <p className="text-xl font-semibold text-white">{queuedCount}</p>
            </div>
            <div className="min-w-[92px] rounded-2xl border border-border bg-secondary/60 px-3 py-2">
              <p className="text-[11px] text-muted-foreground">Completed</p>
              <p className="text-xl font-semibold text-white">{completedCount}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="pointer-events-none absolute right-8 top-1/2 hidden -translate-y-1/2 opacity-20 lg:block">
        <FileVideo className="h-20 w-20 text-muted-foreground" />
      </div>
    </section>
  )
}