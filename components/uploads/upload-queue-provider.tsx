"use client"

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import { cancelVideoUpload, getVideoUploadHistory, getVideoUploadStatus, uploadVideo, type VideoUploadStatus } from "@/lib/api"
import { WalkingLoader } from "@/components/ui/walking-loader"

const MAX_CONCURRENT_UPLOADS = 2
const UPLOAD_QUEUE_STORAGE_KEY = "alive-upload-queue"
const REHYDRATION_POLL_INTERVAL_MS = 1_000

type UploadState = VideoUploadStatus["state"]
type UploadPhase = VideoUploadStatus["phase"]
type UploadJobType = "pedestrian" | "vehicle"

const UPLOAD_STATES: UploadState[] = ["queued", "processing", "complete", "error", "cancelled"]
const UPLOAD_PHASES: NonNullable<UploadPhase>[] = ["queued", "tracking", "vision", "ptsi", "finalizing"]
const CANCELLATION_REQUESTED_MESSAGE_PREFIX = "cancellation requested"
const RECOVERED_INTERRUPTED_UPLOAD_MESSAGE = "This upload was interrupted before completion. Please add the video again."

export interface EnqueuedUploadInput {
  file: File
  locationId: string
  locationName: string
  date: string
  startTime: string
  endTime: string
  fastMode: boolean
  pipeline?: string
  jobType?: UploadJobType
}

export interface UploadQueueItem {
  id: string
  file: File | null
  fileName: string
  fileSize: number
  locationId: string
  locationName: string
  date: string
  startTime: string
  endTime: string
  fastMode: boolean
  pipeline?: string
  jobType?: UploadJobType
  uploadId: string | null
  state: UploadState
  progressPercent: number | null
  message: string
  phase: VideoUploadStatus["phase"]
  videoId: string | null
  error: string | null
  createdAt: string
  updatedAt: string
  startedAt: string | null
  completedAt: string | null
  cancellationRequested: boolean
}

type PersistedUploadQueueItem = Omit<UploadQueueItem, "file">

interface UploadQueueContextValue {
  uploads: UploadQueueItem[]
  enqueueUploads: (items: EnqueuedUploadInput[]) => void
  cancelUpload: (queueItemId: string) => Promise<void>
  activeCount: number
  queuedCount: number
  completedCount: number
  hasActiveUploads: boolean
  settledUploadsVersion: number
  maxConcurrentUploads: number
}

const UploadQueueContext = createContext<UploadQueueContextValue | null>(null)

function createQueueItemId() {
  if (typeof globalThis.crypto !== "undefined" && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID()
  }

  return `queue-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

function isTerminalState(state: UploadState) {
  return state === "complete" || state === "error" || state === "cancelled"
}

function messageRequestsCancellation(message: string | null | undefined) {
  return typeof message === "string" && message.trim().toLowerCase().startsWith(CANCELLATION_REQUESTED_MESSAGE_PREFIX)
}

function isUploadState(value: unknown): value is UploadState {
  return typeof value === "string" && UPLOAD_STATES.includes(value as UploadState)
}

function isUploadPhase(value: unknown): value is UploadPhase {
  return value == null || (typeof value === "string" && UPLOAD_PHASES.includes(value as NonNullable<UploadPhase>))
}

function isActiveUpload(upload: UploadQueueItem) {
  return Boolean(upload.startedAt) && !isTerminalState(upload.state)
}

function isWaitingUpload(upload: UploadQueueItem) {
  return upload.state === "queued" && !upload.startedAt && !upload.cancellationRequested
}

function summarizeUploads(uploads: UploadQueueItem[]) {
  return uploads.reduce(
    (summary, upload) => {
      if (upload.state === "complete") {
        summary.completedCount += 1
        return summary
      }

      if (upload.state === "error" || upload.state === "cancelled") {
        return summary
      }

      const hasStarted =
        Boolean(upload.startedAt) ||
        Boolean(upload.uploadId) ||
        (upload.progressPercent ?? 0) > 0 ||
        Boolean(upload.phase && upload.phase !== "queued")

      if (upload.state === "queued" && !hasStarted && !upload.cancellationRequested) {
        summary.queuedCount += 1
        return summary
      }

      summary.activeCount += 1
      return summary
    },
    { activeCount: 0, queuedCount: 0, completedCount: 0 },
  )
}

function createQueuedItem(input: EnqueuedUploadInput): UploadQueueItem {
  const now = new Date().toISOString()
  const jobType: UploadJobType = input.jobType ?? "pedestrian"

  return {
    id: createQueueItemId(),
    file: input.file,
    fileName: input.file.name,
    fileSize: input.file.size,
    locationId: input.locationId,
    locationName: input.locationName,
    date: input.date,
    startTime: input.startTime,
    endTime: input.endTime,
    fastMode: input.fastMode,
    pipeline: input.pipeline,
    jobType: input.jobType,
    uploadId: null,
    state: "queued",
    progressPercent: 0,
    message: input.fastMode ? "Waiting to upload in fast mode..." : "Waiting to upload...",
    phase: "queued",
    videoId: null,
    error: null,
    createdAt: now,
    updatedAt: now,
    startedAt: null,
    completedAt: null,
    cancellationRequested: false,
  }
}

function createHistoryQueueItemId(uploadId: string) {
  return `history-${uploadId}`
}

function createUploadFromHistory(status: VideoUploadStatus): UploadQueueItem {
  const updatedAt = status.updatedAt
  const createdAt = status.createdAt ?? updatedAt

  return {
    id: createHistoryQueueItemId(status.uploadId),
    file: null,
    fileName: status.fileName ?? `Upload ${status.uploadId.slice(0, 8)}`,
    fileSize: 0,
    locationId: status.locationId ?? "",
    locationName: status.locationName ?? "Unknown location",
    date: status.date ?? "",
    startTime: status.startTime ?? "",
    endTime: status.endTime ?? "",
    fastMode: Boolean(status.fastMode),
    jobType: "pedestrian",
    uploadId: status.uploadId,
    state: status.state,
    progressPercent: typeof status.progressPercent === "number" ? status.progressPercent : null,
    message: status.message,
    phase: status.phase ?? null,
    videoId: status.videoId ?? null,
    error: status.error ?? null,
    createdAt,
    updatedAt,
    startedAt: status.startedAt ?? (status.state === "queued" ? null : createdAt),
    completedAt: status.completedAt ?? (isTerminalState(status.state) ? updatedAt : null),
    cancellationRequested: !isTerminalState(status.state) && messageRequestsCancellation(status.message),
  }
}

function mergeUploadsWithHistory(currentUploads: UploadQueueItem[], history: VideoUploadStatus[]) {
  const historyByUploadId = new Map(history.map((status) => [status.uploadId, status]))
  const seenUploadIds = new Set<string>()
  const recoveredAt = new Date().toISOString()

  const mergedUploads = currentUploads.map((upload) => {
    if (!upload.uploadId) {
      return upload
    }

    const historyStatus = historyByUploadId.get(upload.uploadId)
    if (!historyStatus) {
      if (upload.file || isTerminalState(upload.state)) {
        return upload
      }

      return settleMissingBackendUpload(upload, recoveredAt)
    }

    seenUploadIds.add(upload.uploadId)

    return {
      ...applyStatusToUpload(upload, historyStatus),
      fileName: upload.fileName || historyStatus.fileName || upload.fileName,
      locationId: upload.locationId || historyStatus.locationId || upload.locationId,
      locationName: upload.locationName || historyStatus.locationName || upload.locationName,
      date: upload.date || historyStatus.date || upload.date,
      startTime: upload.startTime || historyStatus.startTime || upload.startTime,
      endTime: upload.endTime || historyStatus.endTime || upload.endTime,
      fastMode: upload.fastMode || Boolean(historyStatus.fastMode),
      jobType: upload.jobType,
      createdAt: upload.createdAt || historyStatus.createdAt || upload.updatedAt,
      startedAt: upload.startedAt ?? historyStatus.startedAt ?? (historyStatus.state === "queued" ? null : historyStatus.updatedAt),
      completedAt: isTerminalState(historyStatus.state)
        ? historyStatus.completedAt ?? historyStatus.updatedAt
        : upload.completedAt,
    }
  })

  for (const historyStatus of history) {
    if (seenUploadIds.has(historyStatus.uploadId)) {
      continue
    }

    mergedUploads.push(createUploadFromHistory(historyStatus))
  }

  return mergedUploads
}

function serializeUpload(upload: UploadQueueItem): PersistedUploadQueueItem {
  const { file: _file, ...persistedUpload } = upload
  return persistedUpload
}

function restoreUpload(value: unknown): UploadQueueItem | null {
  if (!value || typeof value !== "object") {
    return null
  }

  const upload = value as Partial<PersistedUploadQueueItem>
  if (
    typeof upload.id !== "string" ||
    typeof upload.fileName !== "string" ||
    typeof upload.fileSize !== "number" ||
    typeof upload.locationId !== "string" ||
    typeof upload.locationName !== "string" ||
    typeof upload.date !== "string" ||
    typeof upload.startTime !== "string" ||
    typeof upload.endTime !== "string" ||
    typeof upload.message !== "string" ||
    typeof upload.createdAt !== "string" ||
    typeof upload.updatedAt !== "string" ||
    !isUploadState(upload.state)
  ) {
    return null
  }

  return {
    id: upload.id,
    file: null,
    fileName: upload.fileName,
    fileSize: upload.fileSize,
    locationId: upload.locationId,
    locationName: upload.locationName,
    date: upload.date,
    startTime: upload.startTime,
    endTime: upload.endTime,
    fastMode: Boolean(upload.fastMode),
    jobType: upload.jobType === "vehicle" ? "vehicle" : "pedestrian",
    uploadId: typeof upload.uploadId === "string" ? upload.uploadId : null,
    state: upload.state,
    progressPercent: typeof upload.progressPercent === "number" ? upload.progressPercent : null,
    message: upload.message,
    phase: isUploadPhase(upload.phase) ? upload.phase : null,
    videoId: typeof upload.videoId === "string" ? upload.videoId : null,
    error: typeof upload.error === "string" ? upload.error : null,
    createdAt: upload.createdAt,
    updatedAt: upload.updatedAt,
    startedAt: typeof upload.startedAt === "string" ? upload.startedAt : null,
    completedAt: typeof upload.completedAt === "string" ? upload.completedAt : null,
    cancellationRequested: Boolean(upload.cancellationRequested),
  }
}

function applyStatusToUpload(current: UploadQueueItem, status: VideoUploadStatus): UploadQueueItem {
  return {
    ...current,
    uploadId: status.uploadId,
    state: status.state,
    progressPercent: status.progressPercent ?? (status.state === "complete" ? 100 : current.progressPercent),
    message: status.message,
    phase: status.phase ?? current.phase,
    videoId: status.videoId ?? current.videoId,
    error: status.state === "error" ? status.error ?? current.error : null,
    updatedAt: status.updatedAt,
    startedAt: current.startedAt ?? status.startedAt ?? status.updatedAt,
    completedAt: isTerminalState(status.state) ? status.completedAt ?? status.updatedAt : current.completedAt,
    cancellationRequested: !isTerminalState(status.state) && (current.cancellationRequested || messageRequestsCancellation(status.message)),
  }
}

function settleMissingBackendUpload(upload: UploadQueueItem, recoveredAt: string): UploadQueueItem {
  const cancelled = upload.cancellationRequested || messageRequestsCancellation(upload.message)
  const message = cancelled ? "Video upload cancelled." : RECOVERED_INTERRUPTED_UPLOAD_MESSAGE

  return {
    ...upload,
    state: cancelled ? "cancelled" : "error",
    progressPercent: null,
    message,
    phase: null,
    error: cancelled ? null : message,
    updatedAt: recoveredAt,
    completedAt: recoveredAt,
    cancellationRequested: false,
  }
}

function isMissingUploadStatusError(error: unknown) {
  return error instanceof Error && error.message.includes("Upload status not found")
}

function sortUploadsForDisplay(left: UploadQueueItem, right: UploadQueueItem) {
  const rank = (item: UploadQueueItem) => {
    if (!isTerminalState(item.state)) return 0
    if (item.state === "error") return 1
    if (item.state === "cancelled") return 2
    return 3
  }

  const rankDelta = rank(left) - rank(right)
  if (rankDelta !== 0) return rankDelta

  return new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime()
}

export function UploadQueueProvider({ children }: { children: ReactNode }) {
  const [uploads, setUploads] = useState<UploadQueueItem[]>([])
  const [overlayDismissed, setOverlayDismissed] = useState(false)
  const [settledUploadsVersion, setSettledUploadsVersion] = useState(0)
  const uploadsRef = useRef<UploadQueueItem[]>([])
  const launchingIdsRef = useRef(new Set<string>())
  const settledIdsRef = useRef(new Set<string>())
  const hasHydratedRef = useRef(false)

  useEffect(() => {
    if (hasHydratedRef.current || typeof window === "undefined") {
      return
    }

    hasHydratedRef.current = true
    let isCancelled = false
    let parsedUploads: UploadQueueItem[] = []

    try {
      const storedUploads = window.localStorage.getItem(UPLOAD_QUEUE_STORAGE_KEY)
      if (storedUploads) {
        const restoredUploads = JSON.parse(storedUploads)
        if (!Array.isArray(restoredUploads)) {
          window.localStorage.removeItem(UPLOAD_QUEUE_STORAGE_KEY)
        } else {
          parsedUploads = restoredUploads.map(restoreUpload).filter((upload): upload is UploadQueueItem => upload !== null)
        }
      }
    } catch {
      window.localStorage.removeItem(UPLOAD_QUEUE_STORAGE_KEY)
    }

    settledIdsRef.current = new Set(parsedUploads.filter((upload) => isTerminalState(upload.state)).map((upload) => upload.id))
    setUploads(parsedUploads)

    const rehydrateBackendHistory = async () => {
      try {
        const history = await getVideoUploadHistory()
        if (isCancelled || history.length === 0) {
          return
        }

        setUploads((currentUploads) => {
          const mergedUploads = mergeUploadsWithHistory(currentUploads, history)
          settledIdsRef.current = new Set(mergedUploads.filter((upload) => isTerminalState(upload.state)).map((upload) => upload.id))
          return mergedUploads
        })
      } catch {
        // Keep local queue state when backend history is unavailable.
      }
    }

    void rehydrateBackendHistory()

    return () => {
      isCancelled = true
    }
  }, [])

  useEffect(() => {
    uploadsRef.current = uploads
  }, [uploads])

  useEffect(() => {
    if (!hasHydratedRef.current || typeof window === "undefined") {
      return
    }

    if (uploads.length === 0) {
      window.localStorage.removeItem(UPLOAD_QUEUE_STORAGE_KEY)
      return
    }

    window.localStorage.setItem(UPLOAD_QUEUE_STORAGE_KEY, JSON.stringify(uploads.map(serializeUpload)))
  }, [uploads])

  useEffect(() => {
    let discoveredNewTerminalUpload = false

    for (const upload of uploads) {
      if (!isTerminalState(upload.state) || settledIdsRef.current.has(upload.id)) {
        continue
      }

      settledIdsRef.current.add(upload.id)
      discoveredNewTerminalUpload = true
    }

    if (discoveredNewTerminalUpload) {
      setSettledUploadsVersion((current) => current + 1)
    }
  }, [uploads])

  const updateUpload = useCallback((queueItemId: string, updater: (current: UploadQueueItem) => UploadQueueItem) => {
    setUploads((currentUploads) =>
      currentUploads.map((upload) => (upload.id === queueItemId ? updater(upload) : upload)),
    )
  }, [])

  useEffect(() => {
    const strandedUploadIds = uploads
      .filter((upload) => !upload.file && !upload.uploadId && !isTerminalState(upload.state))
      .map((upload) => upload.id)

    if (strandedUploadIds.length === 0) {
      return
    }

    const recoveredAt = new Date().toISOString()
    const strandedUploadIdSet = new Set(strandedUploadIds)

    setUploads((currentUploads) =>
      currentUploads.map((upload) => {
        if (!strandedUploadIdSet.has(upload.id)) {
          return upload
        }

        const message = "This upload needs to be added again because the page was refreshed before it could start."
        return {
          ...upload,
          state: "error",
          error: message,
          message,
          phase: null,
          updatedAt: recoveredAt,
          completedAt: recoveredAt,
        }
      }),
    )
  }, [uploads])

  useEffect(() => {
    const uploadsToSync = uploads.filter(
      (upload) => !upload.file && Boolean(upload.uploadId) && !isTerminalState(upload.state),
    )

    if (uploadsToSync.length === 0 || typeof window === "undefined") {
      return
    }

    let isCancelled = false

    const syncStatuses = async () => {
      await Promise.allSettled(
        uploadsToSync.map(async (upload) => {
          if (!upload.uploadId) {
            return
          }

          try {
            const status = await getVideoUploadStatus(upload.uploadId)
            if (isCancelled) {
              return
            }

            updateUpload(upload.id, (current) => applyStatusToUpload(current, status))
          } catch (error) {
            if (isCancelled || !isMissingUploadStatusError(error)) {
              return
            }

            updateUpload(upload.id, (current) => settleMissingBackendUpload(current, new Date().toISOString()))
          }
        }),
      )
    }

    void syncStatuses()
    const intervalId = window.setInterval(() => {
      void syncStatuses()
    }, REHYDRATION_POLL_INTERVAL_MS)

    return () => {
      isCancelled = true
      window.clearInterval(intervalId)
    }
  }, [updateUpload, uploads])

  const runUpload = useCallback(
    async (queueItemId: string) => {
      const queuedUpload = uploadsRef.current.find((upload) => upload.id === queueItemId)
      if (!queuedUpload || !queuedUpload.file || isTerminalState(queuedUpload.state)) {
        return
      }

      const startedAt = new Date().toISOString()
      updateUpload(queueItemId, (current) => ({
        ...current,
        startedAt: current.startedAt ?? startedAt,
        updatedAt: startedAt,
        message: current.fastMode ? "Preparing fast upload..." : "Preparing upload...",
      }))

      try {
        const result = await uploadVideo({
          file: queuedUpload.file,
          locationId: queuedUpload.locationId,
          date: queuedUpload.date,
          startTime: queuedUpload.startTime,
          endTime: queuedUpload.endTime,
          fastMode: queuedUpload.fastMode,
          pipeline: queuedUpload.pipeline,
          onProgress: (status) => {
            updateUpload(queueItemId, (current) => applyStatusToUpload(current, status))
          },
        })

        updateUpload(queueItemId, (current) => {
          if (isTerminalState(current.state)) {
            return current
          }

          const completedAt = new Date().toISOString()
          return {
            ...current,
            state: "complete",
            progressPercent: 100,
            message: "Video upload and processing complete.",
            videoId: result.id,
            updatedAt: completedAt,
            completedAt,
          }
        })
      } catch (error) {
        const failedAt = new Date().toISOString()
        const errorMessage = error instanceof Error ? error.message : "Failed to upload video."

        updateUpload(queueItemId, (current) => {
          if (isTerminalState(current.state)) {
            return current
          }

          if (current.cancellationRequested) {
            return {
              ...current,
              state: "cancelled",
              message: "Video upload cancelled.",
              updatedAt: failedAt,
              completedAt: failedAt,
            }
          }

          return {
            ...current,
            state: "error",
            error: errorMessage,
            message: errorMessage,
            updatedAt: failedAt,
            completedAt: failedAt,
          }
        })
      }
    },
    [updateUpload],
  )

  useEffect(() => {
    const activeUploads = uploads.filter((upload) => upload.startedAt && !isTerminalState(upload.state)).length
    const availableSlots = MAX_CONCURRENT_UPLOADS - activeUploads
    if (availableSlots <= 0) {
      return
    }

    const pendingUploads = uploads.filter(
      (upload) =>
        Boolean(upload.file) &&
        upload.state === "queued" &&
        !upload.startedAt &&
        !upload.cancellationRequested &&
        !launchingIdsRef.current.has(upload.id),
    )

    pendingUploads.slice(0, availableSlots).forEach((upload) => {
      launchingIdsRef.current.add(upload.id)
      void runUpload(upload.id).finally(() => {
        launchingIdsRef.current.delete(upload.id)
      })
    })
  }, [runUpload, uploads])

  useEffect(() => {
    if (uploads.some((upload) => !isTerminalState(upload.state))) {
      return
    }

    setOverlayDismissed(false)
  }, [uploads])

  const enqueueUploads = useCallback((items: EnqueuedUploadInput[]) => {
    if (items.length === 0) {
      return
    }

    setOverlayDismissed(false)
    setUploads((currentUploads) => [...items.map(createQueuedItem), ...currentUploads])
  }, [])

  const cancelUpload = useCallback(
    async (queueItemId: string) => {
      const upload = uploadsRef.current.find((item) => item.id === queueItemId)
      if (!upload || isTerminalState(upload.state)) {
        return
      }

      const updatedAt = new Date().toISOString()

      if (!upload.startedAt || !upload.uploadId) {
        updateUpload(queueItemId, (current) => ({
          ...current,
          state: "cancelled",
          cancellationRequested: true,
          message: "Video upload cancelled.",
          progressPercent: current.progressPercent ?? 0,
          updatedAt,
          completedAt: updatedAt,
        }))
        return
      }

      updateUpload(queueItemId, (current) => ({
        ...current,
        cancellationRequested: true,
        message: "Cancellation requested. Stopping upload...",
        updatedAt,
      }))

      try {
        const status = await cancelVideoUpload(upload.uploadId)
        updateUpload(queueItemId, (current) => ({
          ...current,
          uploadId: status.uploadId,
          state: status.state,
          progressPercent: status.progressPercent ?? current.progressPercent,
          message: status.message,
          phase: status.phase ?? current.phase,
          videoId: status.videoId ?? current.videoId,
          error: status.error ?? current.error,
          updatedAt: status.updatedAt,
          completedAt: isTerminalState(status.state) ? status.updatedAt : current.completedAt,
          cancellationRequested: true,
        }))
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to cancel upload."
        updateUpload(queueItemId, (current) => ({
          ...current,
          cancellationRequested: false,
          error: message,
          message,
          updatedAt: new Date().toISOString(),
        }))
        throw error
      }
    },
    [updateUpload],
  )

  const sortedUploads = useMemo(() => [...uploads].sort(sortUploadsForDisplay), [uploads])

  const activeUploads = useMemo(() => sortedUploads.filter(isActiveUpload), [sortedUploads])
  const { activeCount, queuedCount, completedCount } = summarizeUploads(sortedUploads)
  const activeOverlayUpload = activeUploads[0] ?? null

  const contextValue = useMemo<UploadQueueContextValue>(
    () => ({
      uploads: sortedUploads,
      enqueueUploads,
      cancelUpload,
      activeCount,
      queuedCount,
      completedCount,
      hasActiveUploads: activeCount > 0,
      settledUploadsVersion,
      maxConcurrentUploads: MAX_CONCURRENT_UPLOADS,
    }),
    [activeCount, cancelUpload, completedCount, enqueueUploads, queuedCount, settledUploadsVersion, sortedUploads],
  )

  return (
    <UploadQueueContext.Provider value={contextValue}>
      {children}
      <WalkingLoader
        isVisible={Boolean(activeOverlayUpload) && !overlayDismissed}
        label={activeOverlayUpload?.message ?? "Uploading video..."}
        progress={activeOverlayUpload?.progressPercent ?? null}
        onClose={() => setOverlayDismissed(true)}
      />
    </UploadQueueContext.Provider>
  )
}

export function useUploadQueue() {
  const context = useContext(UploadQueueContext)
  if (!context) {
    throw new Error("useUploadQueue must be used within an UploadQueueProvider")
  }

  return context
}
