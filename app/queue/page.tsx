"use client"

import Link from "next/link"
import { useState } from "react"
import { QueueItem } from "@/components/queue/queue-item"
import { QueueUploadZone } from "@/components/queue/upload-zone"
import { useUploadQueue } from "@/components/uploads/upload-queue-provider"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { ChevronLeft, FileVideo } from "lucide-react"

export default function QueuePage() {
  const { uploads, cancelUpload, activeCount, queuedCount, completedCount, maxConcurrentUploads } = useUploadQueue()
  const [pendingCancelId, setPendingCancelId] = useState<string | null>(null)
  const [isCancelling, setIsCancelling] = useState(false)
  const pendingCancelUpload = uploads.find((upload) => upload.id === pendingCancelId) ?? null

  const handleConfirmCancel = async () => {
    if (!pendingCancelUpload) {
      return
    }

    try {
      setIsCancelling(true)
      await cancelUpload(pendingCancelUpload.id)
      setPendingCancelId(null)
    } finally {
      setIsCancelling(false)
    }
  }

  return (
    <div className="min-h-full bg-background">
      <header className="border-b border-border bg-card/50 px-6 py-4 backdrop-blur-sm">
        <div className="mx-auto max-w-4xl">
            <div className="mb-4 flex flex-wrap items-center gap-3">
              <Link
                href="/"
                className="inline-flex items-center gap-1 text-sm text-primary transition-colors hover:text-primary/80"
              >
                <ChevronLeft className="h-4 w-4" />
                Back to Pedestrians
              </Link>
              <Link
                href="/vehicles"
                className="inline-flex items-center gap-1 text-sm text-primary transition-colors hover:text-primary/80"
              >
                Back to Vehicles
              </Link>
            </div>

          <div className="flex items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold text-white">Video Processing Queue</h1>
              <p className="mt-1 text-sm text-muted-foreground">
                Manage your video uploads and track processing status.
              </p>
            </div>

            <div className="hidden items-center gap-2 md:flex">
              <div className="rounded-2xl border border-border bg-secondary px-4 py-2 text-right">
                <p className="text-[11px] text-muted-foreground">Total</p>
                <p className="text-lg font-semibold text-white">{uploads.length}</p>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-4xl p-6">
        <QueueUploadZone
          activeCount={activeCount}
          queuedCount={queuedCount}
          completedCount={completedCount}
          maxConcurrentUploads={maxConcurrentUploads}
        />

        <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <div className="rounded-2xl border border-border bg-card px-4 py-3 shadow-elevated-sm">
            <p className="text-sm text-muted-foreground">Active</p>
            <p className="mt-1 text-2xl font-semibold text-white">{activeCount}</p>
          </div>
          <div className="rounded-2xl border border-border bg-card px-4 py-3 shadow-elevated-sm">
            <p className="text-sm text-muted-foreground">Queued</p>
            <p className="mt-1 text-2xl font-semibold text-white">{queuedCount}</p>
          </div>
          <div className="rounded-2xl border border-border bg-card px-4 py-3 shadow-elevated-sm">
            <p className="text-sm text-muted-foreground">Completed</p>
            <p className="mt-1 text-2xl font-semibold text-white">{completedCount}</p>
          </div>
        </div>

        <div className="mt-6 space-y-3">
          {uploads.length === 0 ? (
            <section className="rounded-2xl border border-border bg-card px-6 py-12 text-center shadow-elevated-sm">
              <FileVideo className="mx-auto mb-3 h-12 w-12 opacity-50" />
              <p className="text-muted-foreground">No videos in queue</p>
              <p className="text-sm text-muted-foreground">Upload videos to start processing</p>
            </section>
          ) : (
            uploads.map((upload) => (
              <QueueItem key={upload.id} upload={upload} onCancelRequest={setPendingCancelId} />
            ))
          )}
        </div>
      </div>

      <AlertDialog open={Boolean(pendingCancelUpload)} onOpenChange={(open) => !open && setPendingCancelId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Cancel {pendingCancelUpload?.fileName}?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to cancel this upload? Any incomplete processing for this video will be stopped.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isCancelling}>Keep Uploading</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={isCancelling}
              onClick={(event) => {
                event.preventDefault()
                void handleConfirmCancel()
              }}
            >
              {isCancelling ? "Cancelling..." : "Yes, Cancel Upload"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
