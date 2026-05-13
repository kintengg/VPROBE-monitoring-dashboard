"use client"

import Link from "next/link"
import { useState } from "react"
import { AlertCircle, Check, CheckCircle2, Copy, FileVideo, Loader2, MapPin, Trash2, Video, XCircle } from "lucide-react"
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

function formatInferenceStatusMessage(message: string) {
  const raw = message.trim()
  if (!raw) {
    return raw
  }

  const processingFramesIndex = raw.toLowerCase().indexOf("processing frames:")
  if (processingFramesIndex >= 0) {
    return raw.slice(processingFramesIndex)
  }

  return raw.replace(/^RT-DETR\s+(stdout|stderr):\s*/i, "")
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
      return "Failed"
    case "cancelled":
      return "Upload cancelled"
  }
}

interface QueueItemProps {
  upload: UploadQueueItem
  onCancelRequest: (queueItemId: string) => void
}

export function QueueItem({ upload, onCancelRequest }: QueueItemProps) {
  const [copiedError, setCopiedError] = useState(false)
  const status = getStatusAppearance(upload.state)
  const StatusIcon = status.icon
  const progressValue = upload.state === "complete" ? 100 : upload.progressPercent ?? 0
  const canCancel = upload.state === "queued" || upload.state === "processing"
  const statusText = getStatusText(upload)
  const errorDetails = upload.error ?? upload.message
  const formattedInferenceStatus = formatInferenceStatusMessage(upload.message)

  const handleCopyError = async () => {
    if (!errorDetails || typeof navigator === "undefined" || !navigator.clipboard) {
      return
    }

    try {
      await navigator.clipboard.writeText(errorDetails)
      setCopiedError(true)
      window.setTimeout(() => setCopiedError(false), 1_500)
    } catch {
      // Clipboard access can fail on unsupported browsers or permission denial.
    }
  }

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
                  <Link href={`/video/${upload.videoId}?domain=${upload.domain}`}>
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

          <div className="flex flex-col gap-1 text-sm sm:flex-row sm:items-center sm:justify-between sm:gap-4">
            <span
              className={`min-w-0 break-words ${
                upload.state === "error" || upload.state === "cancelled" ? "text-destructive" : "text-muted-foreground"
              }`}
            >
              {upload.state === "error"
                ? "Processing failed"
                : upload.state === "cancelled"
                  ? "Upload cancelled"
                  : `${upload.startTime} – ${upload.endTime}`}
            </span>
            <span className={`font-medium ${status.tone} sm:shrink-0 sm:text-right`}>
              {statusText}
            </span>
          </div>

          {upload.state !== "error" && formattedInferenceStatus ? (
            <p className="mt-1 text-xs text-muted-foreground">
              Inference status: {formattedInferenceStatus}
            </p>
          ) : null}

          {upload.state === "error" ? (
            <div className="mt-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2">
              <div className="flex items-center justify-between gap-3">
                <p className="text-xs font-medium uppercase tracking-wide text-destructive/90">Error details</p>
                <button
                  type="button"
                  onClick={() => void handleCopyError()}
                  className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-medium text-destructive transition-colors hover:bg-destructive/20"
                  aria-label="Copy error details"
                >
                  {copiedError ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                  {copiedError ? "Copied" : "Copy"}
                </button>
              </div>
              <p className="mt-1 max-h-28 overflow-y-auto whitespace-pre-wrap break-all text-xs leading-relaxed text-destructive">
                {errorDetails || "Unknown error"}
              </p>
            </div>
          ) : null}

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