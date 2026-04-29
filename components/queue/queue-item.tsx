"use client"

import Link from "next/link"
import { AlertCircle, CheckCircle2, FileVideo, Loader2, MapPin, Tractor, Trash2, User, Video, XCircle } from "lucide-react"
import type { UploadQueueItem } from "@/components/uploads/upload-queue-provider"
import { Button } from "@/components/ui/button"

type UploadState = UploadQueueItem["state"]

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

function formatTimestamp(value: string | null) {
  if (!value) return "—"

  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value))
}

function getStatusAppearance(state: UploadState) {
  switch (state) {
    case "queued":
      return {
        tone: "text-primary",
        iconWrap: "bg-primary/20",
        barClass: "bg-primary",
        icon: FileVideo,
      }
    case "processing":
      return {
        tone: "text-chart-4",
        iconWrap: "bg-chart-4/20",
        barClass: "bg-chart-4",
        icon: Loader2,
      }
    case "complete":
      return {
        tone: "text-accent",
        iconWrap: "bg-accent/20",
        barClass: "bg-accent",
        icon: CheckCircle2,
      }
    case "error":
      return {
        tone: "text-destructive",
        iconWrap: "bg-destructive/20",
        barClass: "bg-destructive",
        icon: AlertCircle,
      }
    case "cancelled":
      return {
        tone: "text-muted-foreground",
        iconWrap: "bg-secondary",
        barClass: "bg-muted-foreground",
        icon: XCircle,
      }
  }
}

function getStatusText(upload: UploadQueueItem) {
  switch (upload.state) {
    case "queued":
      return `${Math.round(upload.progressPercent ?? 0)}%`
    case "processing":
      return upload.phase ? `Processing • ${upload.phase}` : "Processing with ALIVE..."
    case "complete":
      return "Processing Complete!"
    case "error":
      return upload.error ?? "Error occurred"
    case "cancelled":
      return "Upload cancelled"
  }
}

interface QueueItemProps {
  upload: UploadQueueItem
  onCancelRequest: (queueItemId: string) => void
}

export function QueueItem({ upload, onCancelRequest }: QueueItemProps) {
  const status = getStatusAppearance(upload.state)
  const StatusIcon = status.icon
  const progressValue = upload.state === "complete" ? 100 : upload.progressPercent ?? 0
  const canCancel = upload.state === "queued" || upload.state === "processing"
  const statusText = getStatusText(upload)
  const isVehicle = upload.jobType === "vehicle"
  const jobBadge = isVehicle
    ? { label: "Vehicle", icon: Tractor, tone: "text-sky-200", bg: "bg-sky-500/15", ring: "ring-sky-500/25" }
    : { label: "Pedestrian", icon: User, tone: "text-emerald-200", bg: "bg-emerald-500/15", ring: "ring-emerald-500/25" }
  const JobIcon = jobBadge.icon

  return (
    <article className="rounded-2xl border border-border bg-card p-4 shadow-elevated-sm transition-all hover:border-border/80">
      <div className="flex items-start gap-4">
        <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${status.iconWrap}`}>
          <StatusIcon className={`h-5 w-5 ${status.tone} ${upload.state === "processing" ? "animate-spin" : ""}`} />
        </div>

        <div className="min-w-0 flex-1">
          <div className="mb-2 flex items-start justify-between gap-4">
            <div className="min-w-0 flex-1">
              <h3 className="truncate pr-4 font-semibold text-white">{upload.fileName}</h3>
              <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
                <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 ${jobBadge.bg} ${jobBadge.tone} ring-1 ${jobBadge.ring}`}>
                  <JobIcon className="h-3 w-3" />
                  {jobBadge.label}
                </span>
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
                <span>{formatBytes(upload.fileSize)}</span>
                <span>•</span>
                <span className="inline-flex items-center gap-1">
                  <MapPin className="h-3 w-3" />
                  {upload.locationName}
                </span>
                <span>•</span>
                <span>{upload.date}</span>
              </div>
            </div>

            <div className="flex shrink-0 items-center gap-2">
              {upload.state === "complete" && upload.videoId ? (
                <Button asChild variant="ghost" size="sm" className="h-8 rounded-xl px-2 text-accent hover:bg-accent/10 hover:text-accent">
                  <Link href={`/video/${upload.videoId}`}>
                    <Video className="h-4 w-4" />
                    Open
                  </Link>
                </Button>
              ) : null}

              {canCancel ? (
                <button
                  type="button"
                  onClick={() => onCancelRequest(upload.id)}
                  disabled={upload.cancellationRequested}
                  className="rounded-lg p-1.5 text-muted-foreground transition-colors hover:bg-destructive/20 hover:text-destructive disabled:cursor-not-allowed disabled:opacity-60"
                  aria-label={upload.cancellationRequested ? "Cancel requested" : "Cancel upload"}
                >
                  <Trash2 className="h-5 w-5" />
                </button>
              ) : upload.state === "complete" ? (
                <div className="rounded-lg bg-accent/20 p-1.5">
                  <CheckCircle2 className="h-5 w-5 text-accent" />
                </div>
              ) : null}
            </div>
          </div>

          <div className="mb-2 h-2 w-full overflow-hidden rounded-full bg-secondary">
            <div
              className={`h-full rounded-full transition-all duration-300 ${status.barClass}`}
              style={{ width: `${Math.max(0, Math.min(progressValue, 100))}%` }}
            />
          </div>

          <div className="flex items-center justify-between gap-4 text-sm">
            <span className={upload.state === "error" || upload.state === "cancelled" ? "text-destructive" : "text-muted-foreground"}>
              {upload.state === "error"
                ? upload.error ?? upload.message
                : upload.state === "cancelled"
                  ? "Upload cancelled"
                  : `${upload.startTime} – ${upload.endTime}${upload.fastMode ? " • Fast mode" : ""}`}
            </span>
            <span className={`text-right font-medium ${status.tone}`}>
              {statusText}
            </span>
          </div>

          <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-muted-foreground">
            <span>Queued {formatTimestamp(upload.createdAt)}</span>
            <span>•</span>
            <span>Updated {formatTimestamp(upload.updatedAt)}</span>
            {upload.phase ? (
              <>
                <span>•</span>
                <span className="capitalize">{upload.phase}</span>
              </>
            ) : null}
            {upload.cancellationRequested ? (
              <>
                <span>•</span>
                <span className="text-destructive">Cancellation requested</span>
              </>
            ) : null}
          </div>
        </div>
      </div>
    </article>
  )
}
