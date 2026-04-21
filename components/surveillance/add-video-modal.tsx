"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { FileVideo, Upload, Video, X, Zap } from "lucide-react"

interface AddVideoModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  locations: Array<{ id: string; name: string }>
  initialLocationId?: string
  onAddVideo?: (data: {
    file: File
    locationId: string
    date: string
    startTime: string
    endTime: string
    fastMode: boolean
  }) => void | Promise<void>
}

const MAX_VISIBLE_FILE_NAME_CHARS = 48

function formatDurationHours(durationSeconds: number) {
  const precision = durationSeconds < 60 ? 5 : durationSeconds < 3600 ? 4 : 2
  return (durationSeconds / 3600).toFixed(precision).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1")
}

function formatHumanDuration(totalSeconds: number) {
  if (totalSeconds < 60) {
    return `${totalSeconds} sec`
  }

  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  const parts = [
    hours > 0 ? `${hours} hr${hours === 1 ? "" : "s"}` : null,
    minutes > 0 ? `${minutes} min` : null,
    seconds > 0 ? `${seconds} sec` : null,
  ].filter(Boolean)

  return parts.join(" ")
}

function formatDisplayFileName(fileName: string, maxChars = MAX_VISIBLE_FILE_NAME_CHARS) {
  if (fileName.length <= maxChars) {
    return fileName
  }

  const extensionIndex = fileName.lastIndexOf(".")
  const hasExtension = extensionIndex > 0 && extensionIndex < fileName.length - 1
  const extension = hasExtension ? fileName.slice(extensionIndex) : ""
  const baseName = hasExtension ? fileName.slice(0, extensionIndex) : fileName
  const availableBaseChars = Math.max(8, maxChars - extension.length - 3)
  const headChars = Math.max(4, Math.ceil(availableBaseChars * 0.6))
  const tailChars = Math.max(3, availableBaseChars - headChars)

  return `${baseName.slice(0, headChars)}...${baseName.slice(-tailChars)}${extension}`
}

function computeSchedule(startTime: string, durationHours: string) {
  const [hoursPart, minutesPart, secondsPart = "0"] = startTime.split(":")
  const startHours = Number(hoursPart)
  const startMinutes = Number(minutesPart)
  const startSeconds = Number(secondsPart)
  const parsedDuration = Number(durationHours)

  if (!startTime || !Number.isFinite(startHours) || !Number.isFinite(startMinutes) || !Number.isFinite(startSeconds) || !Number.isFinite(parsedDuration) || parsedDuration <= 0) {
    return null
  }

  const durationSeconds = Math.max(1, Math.round(parsedDuration * 3600))
  const totalSeconds = startHours * 3600 + startMinutes * 60 + startSeconds + durationSeconds
  const dayOffset = Math.floor(totalSeconds / (24 * 3600))
  const endSeconds = ((totalSeconds % (24 * 3600)) + (24 * 3600)) % (24 * 3600)
  const endHours = Math.floor(endSeconds / 3600)
  const endMinuteValue = Math.floor((endSeconds % 3600) / 60)
  const endSecondValue = endSeconds % 60
  const includeSeconds = endSecondValue > 0 || startTime.split(":").length === 3 || durationSeconds % 60 !== 0

  return {
    endTime: includeSeconds
      ? `${endHours.toString().padStart(2, "0")}:${endMinuteValue.toString().padStart(2, "0")}:${endSecondValue.toString().padStart(2, "0")}`
      : `${endHours.toString().padStart(2, "0")}:${endMinuteValue.toString().padStart(2, "0")}`,
    durationLabel: formatHumanDuration(durationSeconds),
    dayOffset,
  }
}

function readVideoDuration(file: File) {
  return new Promise<number>((resolve, reject) => {
    const video = document.createElement("video")
    const objectUrl = URL.createObjectURL(file)
    let settled = false
    let timeoutId: number | null = null

    const cleanup = () => {
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId)
      }
      video.onloadedmetadata = null
      video.ondurationchange = null
      video.onseeked = null
      video.onerror = null
      URL.revokeObjectURL(objectUrl)
      video.removeAttribute("src")
      video.load()
    }

    const finish = (callback: () => void) => {
      if (settled) {
        return
      }

      settled = true
      cleanup()
      callback()
    }

    const resolveIfDurationReady = () => {
      const duration = Number(video.duration)
      if (Number.isFinite(duration) && duration > 0) {
        finish(() => resolve(duration))
        return true
      }

      return false
    }

    const rejectWithReadableMessage = () => {
      finish(() => reject(new Error("Couldn't auto-read this video's duration in the browser. You can still enter the duration manually.")))
    }

    video.preload = "metadata"
    video.muted = true
    video.playsInline = true
    video.onloadedmetadata = () => {
      if (resolveIfDurationReady()) {
        return
      }

      try {
        video.currentTime = Number.MAX_SAFE_INTEGER
      } catch {
        rejectWithReadableMessage()
      }
    }
    video.ondurationchange = () => {
      resolveIfDurationReady()
    }
    video.onseeked = () => {
      if (!resolveIfDurationReady()) {
        rejectWithReadableMessage()
      }
    }
    video.onerror = () => rejectWithReadableMessage()

    timeoutId = window.setTimeout(() => {
      rejectWithReadableMessage()
    }, 5000)

    video.src = objectUrl
  })
}

export function AddVideoModal({ open, onOpenChange, locations, initialLocationId, onAddVideo }: AddVideoModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [locationId, setLocationId] = useState("")
  const [date, setDate] = useState("")
  const [startTime, setStartTime] = useState("")
  const [durationHours, setDurationHours] = useState("1")
  const [detectedDurationSeconds, setDetectedDurationSeconds] = useState<number | null>(null)
  const [durationError, setDurationError] = useState<string | null>(null)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [isDetectingDuration, setIsDetectingDuration] = useState(false)
  const [fastMode, setFastMode] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setLocationId(initialLocationId ?? "")
      setSubmitError(null)
    }
  }, [initialLocationId, open])

  useEffect(() => {
    if (!selectedFile) {
      setDetectedDurationSeconds(null)
      setDurationError(null)
      setIsDetectingDuration(false)
      return
    }

    let isCancelled = false
    setIsDetectingDuration(true)
    setDurationError(null)

    void readVideoDuration(selectedFile)
      .then((durationSeconds) => {
        if (isCancelled) return
        const roundedSeconds = Math.max(1, Math.round(durationSeconds))
        setDetectedDurationSeconds(roundedSeconds)
        setDurationHours(formatDurationHours(roundedSeconds))
      })
      .catch((error) => {
        if (isCancelled) return
        setDetectedDurationSeconds(null)
        setDurationError(error instanceof Error ? error.message : "Could not read the video duration.")
      })
      .finally(() => {
        if (!isCancelled) {
          setIsDetectingDuration(false)
        }
      })

    return () => {
      isCancelled = true
    }
  }, [selectedFile])

  const computedSchedule = useMemo(() => computeSchedule(startTime, durationHours), [durationHours, startTime])

  const submitDisabledReason = useMemo(() => {
    if (isSubmitting) {
      return "Adding video to the queue..."
    }

    if (isDetectingDuration) {
      return "Reading the selected video before enabling upload..."
    }

    if (!selectedFile) {
      return "Select a video file to continue."
    }

    if (!locationId) {
      return "Choose a location to continue."
    }

    if (!date) {
      return "Choose a start date to continue."
    }

    if (!startTime) {
      return "Choose a start time to continue."
    }

    if (!computedSchedule) {
      return "Enter a valid start time and duration to continue."
    }

    return null
  }, [computedSchedule, date, isDetectingDuration, isSubmitting, locationId, selectedFile, startTime])

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    setSubmitError(null)

    if (e.dataTransfer.files?.[0]) {
      const file = e.dataTransfer.files[0]
      if (file.type === "video/mp4" || file.type === "video/x-msvideo" || file.name.toLowerCase().endsWith(".avi")) {
        setSelectedFile(file)
      }
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) {
      setSubmitError(null)
      setSelectedFile(e.target.files[0])
      e.target.value = ""
    }
  }

  const handleSubmit = async () => {
    if (!selectedFile || !locationId || !date || !startTime || !computedSchedule || isSubmitting) return

    setSubmitError(null)
    setIsSubmitting(true)

    try {
      await onAddVideo?.({
        file: selectedFile,
        locationId,
        date,
        startTime,
        endTime: computedSchedule.endTime,
        fastMode,
      })
      handleClose()
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "Failed to upload video.")
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleClose = () => {
    setSelectedFile(null)
    setLocationId("")
    setDate("")
    setStartTime("")
    setDurationHours("1")
    setDetectedDurationSeconds(null)
    setDurationError(null)
    setSubmitError(null)
    setIsDetectingDuration(false)
    setFastMode(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
    onOpenChange(false)
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) {
          if (isSubmitting) {
            return
          }

          handleClose()
        } else {
          onOpenChange(nextOpen)
        }
      }}
    >
      <DialogContent
        showCloseButton={!isSubmitting}
        className="bg-card border-border sm:max-w-md"
        onEscapeKeyDown={(event) => {
          if (isSubmitting) {
            event.preventDefault()
          }
        }}
        onInteractOutside={(event) => {
          if (isSubmitting) {
            event.preventDefault()
          }
        }}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-foreground">
            <Video className="w-5 h-5" />
            Add New Video
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            Upload a video file, choose the start time, and let the system calculate the end time from the duration.
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          {/* File Upload Area */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Video File</label>
            <div
              className={`relative border-2 border-dashed rounded-lg p-6 transition-colors ${
                dragActive 
                  ? "border-primary bg-primary/5" 
                    : selectedFile
                    ? "border-primary/50 bg-muted/50" 
                    : "border-border hover:border-primary/50"
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".mp4,.avi,video/mp4,video/x-msvideo"
                onChange={handleFileChange}
                className="hidden"
              />

              {selectedFile ? (
                <div className="flex min-w-0 items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                    <FileVideo className="w-5 h-5 text-primary" />
                  </div>
                  <div className="min-w-0 flex-1 overflow-hidden">
                    <p className="overflow-hidden text-ellipsis whitespace-nowrap text-sm font-medium text-foreground" title={selectedFile.name}>
                      {formatDisplayFileName(selectedFile.name)}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                    {isDetectingDuration ? (
                      <p className="text-xs text-primary">Reading duration…</p>
                    ) : detectedDurationSeconds !== null ? (
                      <p className="text-xs text-primary">Detected duration: {formatHumanDuration(detectedDurationSeconds)}</p>
                    ) : durationError ? (
                      <p className="text-xs text-amber-400">{durationError}</p>
                    ) : null}
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => setSelectedFile(null)}
                    disabled={isSubmitting}
                    className="h-8 w-8 shrink-0"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              ) : (
                <div 
                  className="flex flex-col items-center gap-2 cursor-pointer"
                  onClick={() => !isSubmitting && fileInputRef.current?.click()}
                >
                  <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center">
                    <Upload className="w-6 h-6 text-muted-foreground" />
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-medium text-foreground">Drop video here or click to upload</p>
                    <p className="text-xs text-muted-foreground mt-1">Supports MP4 and AVI formats</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Location Dropdown */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Location</label>
            <Select value={locationId} onValueChange={(value) => {
              setSubmitError(null)
              setLocationId(value)
            }}>
              <SelectTrigger className="bg-secondary border-border text-foreground">
                <SelectValue placeholder="Select location" />
              </SelectTrigger>
              <SelectContent className="bg-card border-border">
                {locations.map((loc) => (
                  <SelectItem key={loc.id} value={loc.id} className="text-foreground">
                    {loc.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Date */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Start Date</label>
            <Input 
              type="date" 
              value={date}
              onChange={(e) => {
                setSubmitError(null)
                setDate(e.target.value)
              }}
              className="bg-secondary border-border text-foreground"
            />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Start Time</label>
              <Input 
                type="time" 
                step="1"
                value={startTime}
                onChange={(e) => {
                  setSubmitError(null)
                  setStartTime(e.target.value)
                }}
                className="bg-secondary border-border text-foreground"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Duration (Hours)</label>
              <Input 
                type="number"
                min="0.0003"
                step="0.0001"
                inputMode="decimal"
                value={durationHours}
                onChange={(e) => {
                  setSubmitError(null)
                  setDurationHours(e.target.value)
                }}
                className="bg-secondary border-border text-foreground"
              />
              <p className="text-xs text-muted-foreground">
                {isDetectingDuration
                  ? "Reading the uploaded file to auto-fill the true duration..."
                  : detectedDurationSeconds !== null
                    ? `Auto-filled from video metadata: ${formatHumanDuration(detectedDurationSeconds)}.`
                    : durationError ?? "Upload a file to auto-fill this value, then adjust it manually if needed."}
              </p>
            </div>
          </div>

          <div className="rounded-xl border border-border bg-secondary/40 p-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-foreground">Scheduled coverage</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  {computedSchedule
                    ? `Starts at ${startTime} and ends at ${computedSchedule.endTime}${computedSchedule.dayOffset > 0 ? ` (+${computedSchedule.dayOffset} day)` : ""}. Total duration: ${computedSchedule.durationLabel}.`
                    : "Enter a valid start time and duration to preview the coverage window."}
                </p>
              </div>
              {computedSchedule && <span className="rounded-full bg-primary/10 px-2.5 py-1 text-xs font-medium text-primary">Auto end time</span>}
            </div>
          </div>

          <div className="rounded-xl border border-border bg-secondary/40 p-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                  <Zap className="h-4 w-4 text-primary" />
                  Fast Mode
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  Speeds up local testing by skipping some frames and using a smaller inference size.
                </p>
              </div>
              <Switch checked={fastMode} onCheckedChange={setFastMode} disabled={isSubmitting} className="data-[state=checked]:bg-primary" />
            </div>
          </div>
        </div>

        {submitError && (
          <div className="rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {submitError}
          </div>
        )}

        {!submitError && submitDisabledReason && (
          <p className="text-xs text-muted-foreground">
            {submitDisabledReason}
          </p>
        )}

        <DialogFooter>
          <Button type="button" variant="outline" onClick={handleClose} disabled={isSubmitting} className="border-border text-foreground">
            Cancel
          </Button>
          <Button 
            type="button"
            onClick={() => void handleSubmit()}
            disabled={!selectedFile || !locationId || !date || !startTime || !computedSchedule || isSubmitting || isDetectingDuration}
            className="bg-primary text-primary-foreground hover:bg-primary/90"
          >
            {isSubmitting ? "Adding video..." : isDetectingDuration ? "Reading file..." : "Add Video"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
