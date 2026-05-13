"use client"

import Link from "next/link"
import { useState } from "react"
import { QueueItem } from "@/components/queue/queue-item"
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
import { Button } from "@/components/ui/button"
import { ChevronLeft, FileVideo } from "lucide-react"

export default function QueuePage() {
  const {
    uploads,
    cancelUpload,
    clearHistory,
    activeCount,
    queuedCount,
    completedCount,
    terminalCount,
    maxConcurrentUploads,
    setMaxConcurrentUploads,
  } = useUploadQueue()
  const [pendingCancelId, setPendingCancelId] = useState<string | null>(null)
  const [isCancelling, setIsCancelling] = useState(false)
  const [isClearDialogOpen, setIsClearDialogOpen] = useState(false)
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
          <Link
            href="/"
            className="mb-4 inline-flex items-center gap-1 text-sm text-primary transition-colors hover:text-primary/80"
          >
            <ChevronLeft className="h-4 w-4" />
            Back to Surveillance
          </Link>

          <div className="flex items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold text-white">Video Processing Queue</h1>
              <p className="mt-1 text-sm text-muted-foreground">
                Manage your video uploads and track processing status.
              </p>
            </div>

            <div className="hidden items-center gap-2 md:flex">
              <div className="flex items-center gap-2 rounded-xl border border-border bg-card px-3 py-1.5">
                <span className="text-xs text-muted-foreground">Max concurrent videos</span>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 w-7 rounded-md p-0"
                  disabled={maxConcurrentUploads <= 1}
                  onClick={() => setMaxConcurrentUploads(maxConcurrentUploads - 1)}
                >
                  -
                </Button>
                <input
                  type="number"
                  min="1"
                  value={maxConcurrentUploads}
                  onChange={(e) => setMaxConcurrentUploads(parseInt(e.target.value, 10) || 1)}
                  className="w-12 bg-transparent text-center text-sm font-semibold text-white outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
                />
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-7 w-7 rounded-md p-0"
                  onClick={() => setMaxConcurrentUploads(maxConcurrentUploads + 1)}
                >
                  +
                </Button>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="rounded-xl"
                disabled={terminalCount === 0}
                onClick={() => setIsClearDialogOpen(true)}
              >
                Clear Queue
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-4xl p-6">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
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
          <div className="rounded-2xl border border-border bg-card px-4 py-3 shadow-elevated-sm">
            <p className="text-sm text-muted-foreground">Total</p>
            <p className="mt-1 text-2xl font-semibold text-white">{uploads.length}</p>
          </div>
        </div>

        <div className="mt-6 space-y-3">
          {uploads.length === 0 ? (
            <section className="rounded-2xl border border-border bg-card px-6 py-12 text-center shadow-elevated-sm">
              <FileVideo className="mx-auto mb-3 h-12 w-12 opacity-50" />
              <p className="text-muted-foreground">No videos in queue</p>
              <p className="text-sm text-muted-foreground">Queue items will appear here while videos are being processed.</p>
            </section>
          ) : (
            uploads.map((upload) => (
              <QueueItem key={upload.id} upload={upload} onCancelRequest={setPendingCancelId} />
            ))
          )}
        </div>

        <div className="mt-4 flex md:hidden">
          <div className="flex w-full items-center gap-2">
            <div className="flex flex-1 items-center justify-center gap-2 rounded-xl border border-border bg-card px-3 py-1.5">
              <span className="text-xs text-muted-foreground">Max concurrent videos</span>
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-7 w-7 rounded-md p-0"
                disabled={maxConcurrentUploads <= 1}
                onClick={() => setMaxConcurrentUploads(maxConcurrentUploads - 1)}
              >
                -
              </Button>
              <input
                type="number"
                min="1"
                value={maxConcurrentUploads}
                onChange={(e) => setMaxConcurrentUploads(parseInt(e.target.value, 10) || 1)}
                className="w-12 bg-transparent text-center text-sm font-semibold text-white outline-none [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-7 w-7 rounded-md p-0"
                onClick={() => setMaxConcurrentUploads(maxConcurrentUploads + 1)}
              >
                +
              </Button>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="rounded-xl"
              disabled={terminalCount === 0}
              onClick={() => setIsClearDialogOpen(true)}
            >
              Clear Queue
            </Button>
          </div>
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

      <AlertDialog open={isClearDialogOpen} onOpenChange={setIsClearDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear completed uploads?</AlertDialogTitle>
            <AlertDialogDescription>
              This clears finished items from the list. Videos that are still uploading or processing will stay.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep Items</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={(event) => {
                event.preventDefault()
                clearHistory()
                setIsClearDialogOpen(false)
              }}
            >
              Clear Queue
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}