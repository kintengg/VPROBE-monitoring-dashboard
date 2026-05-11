"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
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
import { FileVideo, Upload, Video, X } from "lucide-react"
import { getCountingConfigChoices, uploadInferenceRequirement } from "@/lib/api"

const LAST_COUNTING_CONFIG_STORAGE_KEY = "alive-last-counting-config"
const LAST_VIDEO_FORM_VALUES_STORAGE_KEY = "alive-last-video-form-values"
const LAST_GATE_LOS_VALUES_STORAGE_KEY = "alive-last-gate-los-values"

const GATE_DEFAULT_LOS_VALUES: Record<string, { roadLengthM: number; laneCount: number }> = {
  g2: { roadLengthM: 26.3, laneCount: 3 },
  "g2.9": { roadLengthM: 60, laneCount: 2 },
  g3: { roadLengthM: 45.3, laneCount: 3 },
  "g3.2": { roadLengthM: 20, laneCount: 2 },
  "g3.5": { roadLengthM: 20.6, laneCount: 1 },
}

const HOUR_OPTIONS = Array.from({ length: 12 }, (_, index) => {
  const hour = index + 1
  return { value: String(hour).padStart(2, "0"), label: String(hour) }
})

const MINUTE_SECOND_OPTIONS = Array.from({ length: 60 }, (_, index) => {
  const value = String(index).padStart(2, "0")
  return { value, label: value }
})

const MERIDIEM_OPTIONS = ["AM", "PM"] as const
const MONTH_OPTIONS = [
  { value: "01", label: "Jan" },
  { value: "02", label: "Feb" },
  { value: "03", label: "Mar" },
  { value: "04", label: "Apr" },
  { value: "05", label: "May" },
  { value: "06", label: "Jun" },
  { value: "07", label: "Jul" },
  { value: "08", label: "Aug" },
  { value: "09", label: "Sep" },
  { value: "10", label: "Oct" },
  { value: "11", label: "Nov" },
  { value: "12", label: "Dec" },
]
const DAY_OPTIONS = Array.from({ length: 31 }, (_, index) => {
  const day = String(index + 1).padStart(2, "0")
  return { value: day, label: day }
})
const MIN_YEAR_OPTION = 2000
const MAX_YEAR_OPTION = 2100
const YEAR_OPTIONS = Array.from({ length: MAX_YEAR_OPTION - MIN_YEAR_OPTION + 1 }, (_, index) => {
  const year = String(MIN_YEAR_OPTION + index)
  return { value: year, label: year }
})

function inferGateSuffixFromLocation(locationName: string): string | null {
  const normalizedLocation = locationName.toLowerCase().trim()
  const compactLocation = normalizedLocation.replace(/[^a-z0-9.]+/g, "")

  const hasGateKeyword = /\bgate\b/.test(normalizedLocation) || /\bgate\s*\d/.test(normalizedLocation) || compactLocation.startsWith("gate")
  const hasGPrefix = /^g\d/.test(compactLocation)
  if (!hasGateKeyword && !hasGPrefix) {
    return null
  }

  let gateToken: string | null = null
  if (hasGPrefix) {
    const compactMatch = compactLocation.match(/^g(\d\.\d|\d{2}|\d)/)
    gateToken = compactMatch?.[1] ?? null
  }

  if (!gateToken && hasGateKeyword) {
    const gateMatch = normalizedLocation.match(/\bgate\s*(\d\.\d|\d{2}|\d)\b/)
    gateToken = gateMatch?.[1] ?? null

    if (!gateToken) {
      const compactGateMatch = compactLocation.match(/gate(\d\.\d|\d{2}|\d)/)
      gateToken = compactGateMatch?.[1] ?? null
    }
  }

  if (!gateToken) {
    return null
  }

  return {
    "2": "g2",
    "29": "g2.9",
    "2.9": "g2.9",
    "3": "g3",
    "32": "g3.2",
    "3.2": "g3.2",
    "35": "g3.5",
    "3.5": "g3.5",
  }[gateToken] ?? null
}

function inferGateSuffixFromLocationId(locationId: string): string | null {
  const normalized = locationId.trim().toLowerCase()
  return {
    "gate-2": "g2",
    "gate-2-9": "g2.9",
    "gate-3": "g3",
    "gate-3-2": "g3.2",
    "gate-3-5": "g3.5",
  }[normalized] ?? null
}

function findGateMatchedCountingConfig(options: string[], locationName: string): string | null {
  const suffix = inferGateSuffixFromLocation(locationName)
  if (!suffix) {
    return null
  }

  const expectedFileName = `counting_config_${suffix}.json`
  const exact = options.find((option) => option.toLowerCase() === expectedFileName.toLowerCase())
  if (exact) {
    return exact
  }

  const escapedSuffix = suffix.replace(".", "\\.")
  const suffixPattern = new RegExp(`(^|[^a-z0-9])${escapedSuffix}([^a-z0-9]|$)`, "i")
  return options.find((option) => suffixPattern.test(option.toLowerCase())) ?? null
}

function composeTwentyFourHourTime(hour12: string, minute: string, second: string, meridiem: string) {
  const parsedHour12 = Number(hour12)
  if (!Number.isFinite(parsedHour12) || parsedHour12 < 1 || parsedHour12 > 12) {
    return ""
  }

  const parsedMinute = Number(minute)
  const parsedSecond = Number(second)
  if (!Number.isFinite(parsedMinute) || parsedMinute < 0 || parsedMinute > 59) {
    return ""
  }
  if (!Number.isFinite(parsedSecond) || parsedSecond < 0 || parsedSecond > 59) {
    return ""
  }

  const normalizedMeridiem = meridiem === "PM" ? "PM" : meridiem === "AM" ? "AM" : ""
  if (!normalizedMeridiem) {
    return ""
  }

  const hour24 = normalizedMeridiem === "PM"
    ? (parsedHour12 % 12) + 12
    : parsedHour12 % 12

  return `${String(hour24).padStart(2, "0")}:${String(parsedMinute).padStart(2, "0")}:${String(parsedSecond).padStart(2, "0")}`
}

function composeIsoDate(year: string, month: string, day: string) {
  const parsedYear = Number(year)
  const parsedMonth = Number(month)
  const parsedDay = Number(day)
  if (!Number.isInteger(parsedYear) || parsedYear < 1) {
    return ""
  }
  if (!Number.isInteger(parsedMonth) || parsedMonth < 1 || parsedMonth > 12) {
    return ""
  }
  if (!Number.isInteger(parsedDay) || parsedDay < 1 || parsedDay > 31) {
    return ""
  }

  const candidate = new Date(parsedYear, parsedMonth - 1, parsedDay)
  if (
    candidate.getFullYear() !== parsedYear ||
    candidate.getMonth() !== parsedMonth - 1 ||
    candidate.getDate() !== parsedDay
  ) {
    return ""
  }

  return `${year}-${month}-${day}`
}

interface AddVideoModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  locations: Array<{ id: string; name: string; roadLengthM?: number | null; laneCount?: number | null }>
  initialLocationId?: string
  onAddVideo?: (data: {
    file: File
    locationId: string
    date: string
    startTime: string
    roadLengthM?: number
    laneCount?: number
    countingConfig?: string
    showLivePreview?: boolean
  }) => void | Promise<void>
}

interface LastVideoFormValues {
  locationId: string
  date: string
  startHour: string
  startMinute: string
  startSecond: string
  startMeridiem: string
  selectedCountingConfig: string
  showLivePreview: boolean
}

interface LastGateLosValues {
  [gateSuffix: string]: {
    roadLengthM: number
    laneCount: number
  }
}

function parseIsoDateParts(value: string) {
  const match = value.match(/^(\d{4})-(\d{2})-(\d{2})$/)
  if (!match) {
    return { year: "", month: "", day: "" }
  }

  return {
    year: match[1],
    month: match[2],
    day: match[3],
  }
}

function readLastVideoFormValues(): LastVideoFormValues | null {
  if (typeof window === "undefined") {
    return null
  }

  const raw = window.localStorage.getItem(LAST_VIDEO_FORM_VALUES_STORAGE_KEY)
  if (!raw) {
    return null
  }

  try {
    const parsed = JSON.parse(raw) as Partial<LastVideoFormValues>
    return {
      locationId: typeof parsed.locationId === "string" ? parsed.locationId : "",
      date: typeof parsed.date === "string" ? parsed.date : "",
      startHour: typeof parsed.startHour === "string" ? parsed.startHour : "",
      startMinute: typeof parsed.startMinute === "string" ? parsed.startMinute : "",
      startSecond: typeof parsed.startSecond === "string" ? parsed.startSecond : "",
      startMeridiem: parsed.startMeridiem === "AM" || parsed.startMeridiem === "PM" ? parsed.startMeridiem : "",
      selectedCountingConfig: typeof parsed.selectedCountingConfig === "string" ? parsed.selectedCountingConfig : "",
      showLivePreview: Boolean(parsed.showLivePreview),
    }
  } catch {
    return null
  }
}

function writeLastVideoFormValues(values: LastVideoFormValues) {
  if (typeof window === "undefined") {
    return
  }

  window.localStorage.setItem(LAST_VIDEO_FORM_VALUES_STORAGE_KEY, JSON.stringify(values))
}

function readLastGateLosValues(): LastGateLosValues {
  if (typeof window === "undefined") {
    return {}
  }

  try {
    const raw = window.localStorage.getItem(LAST_GATE_LOS_VALUES_STORAGE_KEY)
    if (!raw) {
      return {}
    }

    const parsed = JSON.parse(raw) as Record<string, unknown>
    const normalized: LastGateLosValues = {}

    for (const [key, value] of Object.entries(parsed)) {
      if (!value || typeof value !== "object") {
        continue
      }

      const record = value as { roadLengthM?: unknown; laneCount?: unknown }
      const roadLengthM = Number(record.roadLengthM)
      const laneCount = Number(record.laneCount)

      if (Number.isFinite(roadLengthM) && roadLengthM > 0 && Number.isFinite(laneCount) && laneCount > 0) {
        normalized[key] = {
          roadLengthM,
          laneCount: Math.round(laneCount),
        }
      }
    }

    return normalized
  } catch {
    return {}
  }
}

function writeLastGateLosValues(values: LastGateLosValues) {
  if (typeof window === "undefined") {
    return
  }

  window.localStorage.setItem(LAST_GATE_LOS_VALUES_STORAGE_KEY, JSON.stringify(values))
}

function resolveGateLosDefaults(location: { id: string; name: string; roadLengthM?: number | null; laneCount?: number | null } | undefined) {
  if (!location) {
    return { roadLengthM: "", laneCount: "" }
  }

  const gateSuffix = inferGateSuffixFromLocationId(location.id) ?? inferGateSuffixFromLocation(location.name)

  if (gateSuffix && GATE_DEFAULT_LOS_VALUES[gateSuffix]) {
    return {
      roadLengthM: String(GATE_DEFAULT_LOS_VALUES[gateSuffix].roadLengthM),
      laneCount: String(GATE_DEFAULT_LOS_VALUES[gateSuffix].laneCount),
    }
  }

  return { roadLengthM: "", laneCount: "" }
}

const ACCEPTED_VIDEO_MIME_TYPES = new Set(["video/mp4", "video/x-msvideo", "video/avi", "video/msvideo"])

function isAcceptedVideoFile(file: File) {
  const normalizedName = file.name.toLowerCase()
  if (normalizedName.endsWith(".mp4") || normalizedName.endsWith(".avi")) {
    return true
  }

  return ACCEPTED_VIDEO_MIME_TYPES.has(file.type.toLowerCase())
}

export function AddVideoModal({ open, onOpenChange, locations, initialLocationId, onAddVideo }: AddVideoModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [locationId, setLocationId] = useState("")
  const [date, setDate] = useState("")
  const [startYear, setStartYear] = useState("")
  const [startMonth, setStartMonth] = useState("")
  const [startDay, setStartDay] = useState("")
  const [startTime, setStartTime] = useState("")
  const [startHour, setStartHour] = useState("")
  const [startMinute, setStartMinute] = useState("")
  const [startSecond, setStartSecond] = useState("")
  const [startMeridiem, setStartMeridiem] = useState("")
  const [roadLengthM, setRoadLengthM] = useState("")
  const [laneCount, setLaneCount] = useState("")
  const [countingOptions, setCountingOptions] = useState<string[]>([])
  const [selectedCountingConfig, setSelectedCountingConfig] = useState("")
  const [countingConfigError, setCountingConfigError] = useState<string | null>(null)
  const [isLoadingCountingOptions, setIsLoadingCountingOptions] = useState(false)
  const [isUploadingCountingConfig, setIsUploadingCountingConfig] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [showLivePreview, setShowLivePreview] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const countingConfigInputRef = useRef<HTMLInputElement>(null)

  const refreshCountingOptions = useCallback(async () => {
    setIsLoadingCountingOptions(true)
    setCountingConfigError(null)

    try {
      const response = await getCountingConfigChoices()
      const nextOptions = response.options ?? []
      const savedSelection = typeof window !== "undefined" ? window.localStorage.getItem(LAST_COUNTING_CONFIG_STORAGE_KEY) : null
      const fallbackSelection = response.defaultConfig && nextOptions.includes(response.defaultConfig) ? response.defaultConfig : nextOptions[0] ?? ""
      const resolvedSelection = savedSelection && nextOptions.includes(savedSelection) ? savedSelection : fallbackSelection

      setCountingOptions(nextOptions)
      setSelectedCountingConfig((current) => {
        if (current && nextOptions.includes(current)) {
          return current
        }
        return resolvedSelection
      })
    } catch (error) {
      setCountingConfigError(error instanceof Error ? error.message : "Failed to load counting configs.")
      setCountingOptions([])
      setSelectedCountingConfig("")
    } finally {
      setIsLoadingCountingOptions(false)
    }
  }, [])

  useEffect(() => {
    if (open) {
      const savedValues = readLastVideoFormValues()
      const normalizedInitialLocationId = typeof initialLocationId === "string" && initialLocationId.trim().length > 0
        ? initialLocationId
        : undefined
      const resolvedLocationId = normalizedInitialLocationId ?? savedValues?.locationId ?? ""
      const resolvedDate = savedValues?.date ?? ""
      const resolvedHour = savedValues?.startHour ?? ""
      const resolvedMinute = savedValues?.startMinute ?? ""
      const resolvedSecond = savedValues?.startSecond ?? ""
      const resolvedMeridiem = savedValues?.startMeridiem ?? ""
      const resolvedStartTime = composeTwentyFourHourTime(resolvedHour, resolvedMinute, resolvedSecond, resolvedMeridiem)

      setLocationId(resolvedLocationId)
      setDate(resolvedDate)
      const dateParts = parseIsoDateParts(resolvedDate)
      setStartYear(dateParts.year)
      setStartMonth(dateParts.month)
      setStartDay(dateParts.day)
      setStartHour(resolvedHour)
      setStartMinute(resolvedMinute)
      setStartSecond(resolvedSecond)
      setStartMeridiem(resolvedMeridiem)
      setStartTime(resolvedStartTime)
      const selectedLocation = locations.find((location) => location.id === resolvedLocationId)
      const defaults = resolveGateLosDefaults(selectedLocation)
      setRoadLengthM(defaults.roadLengthM)
      setLaneCount(defaults.laneCount)
      setShowLivePreview(Boolean(savedValues?.showLivePreview))
      if (savedValues?.selectedCountingConfig) {
        setSelectedCountingConfig(savedValues.selectedCountingConfig)
      }
      setSubmitError(null)
      void refreshCountingOptions()
    }
  }, [initialLocationId, locations, open, refreshCountingOptions])

  const setClockPart = useCallback(
    (next: { hour?: string; minute?: string; second?: string; meridiem?: string }) => {
      const nextHour = next.hour ?? startHour
      const nextMinute = next.minute ?? startMinute
      const nextSecond = next.second ?? startSecond
      const nextMeridiem = next.meridiem ?? startMeridiem
      const nextTime = composeTwentyFourHourTime(nextHour, nextMinute, nextSecond, nextMeridiem)
      setStartTime(nextTime)
    },
    [startHour, startMeridiem, startMinute, startSecond],
  )

  const setDatePart = useCallback(
    (next: { year?: string; month?: string; day?: string }) => {
      const nextYear = next.year ?? startYear
      const nextMonth = next.month ?? startMonth
      const nextDay = next.day ?? startDay
      const nextDate = composeIsoDate(nextYear, nextMonth, nextDay)
      setDate(nextDate)
    },
    [startDay, startMonth, startYear],
  )

  useEffect(() => {
    if (selectedCountingConfig && typeof window !== "undefined") {
      window.localStorage.setItem(LAST_COUNTING_CONFIG_STORAGE_KEY, selectedCountingConfig)
    }
  }, [selectedCountingConfig])

  useEffect(() => {
    if (!open) {
      return
    }

    writeLastVideoFormValues({
      locationId,
      date,
      startHour,
      startMinute,
      startSecond,
      startMeridiem,
      selectedCountingConfig,
      showLivePreview,
    })
  }, [date, locationId, open, selectedCountingConfig, showLivePreview, startHour, startMeridiem, startMinute, startSecond])

  useEffect(() => {
    if (!locationId || countingOptions.length === 0) {
      return
    }

    const locationName = locations.find((location) => location.id === locationId)?.name
    if (!locationName) {
      return
    }

    const matched = findGateMatchedCountingConfig(countingOptions, locationName)
    if (matched && matched !== selectedCountingConfig) {
      setSelectedCountingConfig(matched)
      setCountingConfigError(null)
    }
  }, [countingOptions, locationId, locations, selectedCountingConfig])

  useEffect(() => {
    if (!open || !locationId) {
      return
    }

    const selectedLocation = locations.find((location) => location.id === locationId)
    const defaults = resolveGateLosDefaults(selectedLocation)
    setRoadLengthM(defaults.roadLengthM)
    setLaneCount(defaults.laneCount)
  }, [locationId, locations, open])

  const submitDisabledReason = useMemo(() => {
    if (isSubmitting) {
      return "Adding video to the queue..."
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

    if (!selectedCountingConfig) {
      return "Choose a counting config to continue."
    }

    if ((roadLengthM.trim() && !laneCount.trim()) || (!roadLengthM.trim() && laneCount.trim())) {
      return "Provide both road length and lane count together."
    }

    return null
  }, [date, isSubmitting, laneCount, locationId, roadLengthM, selectedCountingConfig, selectedFile, startTime])

  const formattedDateLabel = useMemo(() => {
    if (!date) {
      return "Select month, day, and year."
    }

    const parsed = new Date(`${date}T00:00:00`)
    if (Number.isNaN(parsed.getTime())) {
      return "Select month, day, and year."
    }

    return parsed.toLocaleDateString("en-US", {
      weekday: "short",
      month: "short",
      day: "numeric",
      year: "numeric",
    })
  }, [date])

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
      if (isAcceptedVideoFile(file)) {
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

  const handleCountingConfigUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) {
      return
    }

    if (!file.name.toLowerCase().endsWith(".json")) {
      setCountingConfigError("Counting config upload must be a .json file.")
      return
    }

    setIsUploadingCountingConfig(true)
    setCountingConfigError(null)

    try {
      const result = await uploadInferenceRequirement({
        file,
        requirementType: "counting-config",
      })
      await refreshCountingOptions()
      setSelectedCountingConfig(result.filename)
    } catch (error) {
      setCountingConfigError(error instanceof Error ? error.message : "Failed to upload counting config.")
    } finally {
      setIsUploadingCountingConfig(false)
    }
  }

  const handleSubmit = async () => {
    if (!selectedFile || !locationId || !date || !startTime || !selectedCountingConfig || isSubmitting) return

    let parsedRoadLengthM: number | undefined
    let parsedLaneCount: number | undefined

    if (roadLengthM.trim() || laneCount.trim()) {
      if (!roadLengthM.trim() || !laneCount.trim()) {
        setSubmitError("Provide both road length and lane count together.")
        return
      }

      const roadLength = Number.parseFloat(roadLengthM)
      const lanes = Number.parseInt(laneCount, 10)

      if (!Number.isFinite(roadLength) || roadLength <= 0) {
        setSubmitError("Road length must be a positive number.")
        return
      }

      if (!Number.isFinite(lanes) || lanes <= 0) {
        setSubmitError("Lane count must be a positive whole number.")
        return
      }

      parsedRoadLengthM = roadLength
      parsedLaneCount = lanes

      const selectedLocation = locations.find((location) => location.id === locationId)
      const gateSuffix = selectedLocation
        ? (inferGateSuffixFromLocationId(selectedLocation.id) ?? inferGateSuffixFromLocation(selectedLocation.name))
        : null

      if (gateSuffix) {
        const existingValues = readLastGateLosValues()
        existingValues[gateSuffix] = { roadLengthM: roadLength, laneCount: lanes }
        writeLastGateLosValues(existingValues)
      }
    }

    setSubmitError(null)
    setIsSubmitting(true)

    try {
      await onAddVideo?.({
        file: selectedFile,
        locationId,
        date,
        startTime,
        roadLengthM: parsedRoadLengthM,
        laneCount: parsedLaneCount,
        countingConfig: selectedCountingConfig,
        showLivePreview,
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
    setCountingConfigError(null)
    setSubmitError(null)
    setIsLoadingCountingOptions(false)
    setIsUploadingCountingConfig(false)
    setShowLivePreview(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
    if (countingConfigInputRef.current) {
      countingConfigInputRef.current.value = ""
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
        className="bg-card border-border sm:max-w-2xl"
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
            Upload a video file and choose start metadata. End time is computed automatically after processing.
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-2.5 py-2.5">
          {/* File Upload Area */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Video File</label>
            <div
              className={`relative rounded-lg border-2 border-dashed p-3 transition-colors ${
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
                <div className="flex items-center gap-2">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                    <FileVideo className="h-4 w-4 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="truncate text-sm font-medium text-foreground">{selectedFile.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => setSelectedFile(null)}
                    disabled={isSubmitting}
                    className="h-8 w-8"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              ) : (
                <div 
                  className="flex flex-col items-center gap-2 cursor-pointer"
                  onClick={() => !isSubmitting && fileInputRef.current?.click()}
                >
                  <div className="flex h-9 w-9 items-center justify-center rounded-full bg-muted">
                    <Upload className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="text-center">
                    <p className="text-xs font-medium text-foreground">Drop video here or click to upload</p>
                    <p className="mt-0.5 text-[11px] text-muted-foreground">Supports MP4 and AVI formats</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-foreground">Location</label>
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
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-foreground">Road Length (m)</label>
              <Input
                type="number"
                step="0.01"
                min="0"
                value={roadLengthM}
                onChange={(event) => {
                  setSubmitError(null)
                  setRoadLengthM(event.target.value)
                }}
                placeholder="e.g., 18"
                disabled={isSubmitting}
                className="bg-secondary border-border text-foreground"
              />
            </div>
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-foreground">Lane Count</label>
              <Input
                type="number"
                step="1"
                min="1"
                value={laneCount}
                onChange={(event) => {
                  setSubmitError(null)
                  setLaneCount(event.target.value)
                }}
                placeholder="e.g., 2"
                disabled={isSubmitting}
                className="bg-secondary border-border text-foreground"
              />
            </div>
          </div>


          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <label className="text-sm font-medium text-foreground">Counting Config</label>
              <input
                ref={countingConfigInputRef}
                type="file"
                accept=".json,application/json"
                onChange={(event) => {
                  void handleCountingConfigUpload(event)
                }}
                className="hidden"
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="border-border"
                disabled={isSubmitting || isUploadingCountingConfig}
                onClick={() => countingConfigInputRef.current?.click()}
              >
                {isUploadingCountingConfig ? "Uploading..." : "Upload New"}
              </Button>
            </div>
            <Select
              value={selectedCountingConfig}
              onValueChange={(value) => {
                setSubmitError(null)
                setCountingConfigError(null)
                setSelectedCountingConfig(value)
              }}
              disabled={isLoadingCountingOptions || isSubmitting}
            >
              <SelectTrigger className="bg-secondary border-border text-foreground">
                <SelectValue placeholder={isLoadingCountingOptions ? "Loading counting configs..." : "Select counting config"} />
              </SelectTrigger>
              <SelectContent className="bg-card border-border">
                {countingOptions.map((configName) => (
                  <SelectItem key={configName} value={configName} className="text-foreground">
                    {configName}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-[11px] text-muted-foreground">
              Choose an existing counting-line config or upload a new .json file.
            </p>
            {countingConfigError ? <p className="text-xs text-destructive">{countingConfigError}</p> : null}
          </div>

          {/* Date */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Start Date</label>
            <div className="grid grid-cols-3 gap-2 sm:gap-3">
              <Select
                value={startMonth}
                onValueChange={(value) => {
                  setSubmitError(null)
                  setStartMonth(value)
                  setDatePart({ month: value })
                }}
                disabled={isSubmitting}
              >
                <SelectTrigger className="w-full bg-secondary border-border text-foreground">
                  <SelectValue placeholder="Month" />
                </SelectTrigger>
                <SelectContent className="max-h-72 bg-card border-border">
                  {MONTH_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value} className="text-foreground">
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={startDay}
                onValueChange={(value) => {
                  setSubmitError(null)
                  setStartDay(value)
                  setDatePart({ day: value })
                }}
                disabled={isSubmitting}
              >
                <SelectTrigger className="w-full bg-secondary border-border text-foreground">
                  <SelectValue placeholder="Day" />
                </SelectTrigger>
                <SelectContent className="max-h-72 bg-card border-border">
                  {DAY_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value} className="text-foreground">
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={startYear}
                onValueChange={(value) => {
                  setSubmitError(null)
                  setStartYear(value)
                  setDatePart({ year: value })
                }}
                disabled={isSubmitting}
              >
                <SelectTrigger className="w-full bg-secondary border-border text-foreground">
                  <SelectValue placeholder="Year" />
                </SelectTrigger>
                <SelectContent className="max-h-72 bg-card border-border">
                  {YEAR_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value} className="text-foreground">
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <p className="text-[11px] text-muted-foreground">{formattedDateLabel}</p>
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Start Time</label>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,110px)] sm:gap-3">
              <Select
                value={startHour}
                onValueChange={(value) => {
                  setSubmitError(null)
                  setStartHour(value)
                  setClockPart({ hour: value })
                }}
                disabled={isSubmitting}
              >
                <SelectTrigger className="w-full bg-secondary border-border text-foreground">
                  <SelectValue placeholder="HH" />
                </SelectTrigger>
                <SelectContent className="max-h-72 bg-card border-border">
                  {HOUR_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value} className="text-foreground">
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={startMinute}
                onValueChange={(value) => {
                  setSubmitError(null)
                  setStartMinute(value)
                  setClockPart({ minute: value })
                }}
                disabled={isSubmitting}
              >
                <SelectTrigger className="w-full bg-secondary border-border text-foreground">
                  <SelectValue placeholder="MM" />
                </SelectTrigger>
                <SelectContent className="max-h-72 bg-card border-border">
                  {MINUTE_SECOND_OPTIONS.map((option) => (
                    <SelectItem key={`minute-${option.value}`} value={option.value} className="text-foreground">
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={startSecond}
                onValueChange={(value) => {
                  setSubmitError(null)
                  setStartSecond(value)
                  setClockPart({ second: value })
                }}
                disabled={isSubmitting}
              >
                <SelectTrigger className="w-full bg-secondary border-border text-foreground">
                  <SelectValue placeholder="SS" />
                </SelectTrigger>
                <SelectContent className="max-h-72 bg-card border-border">
                  {MINUTE_SECOND_OPTIONS.map((option) => (
                    <SelectItem key={`second-${option.value}`} value={option.value} className="text-foreground">
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select
                value={startMeridiem}
                onValueChange={(value) => {
                  setSubmitError(null)
                  setStartMeridiem(value)
                  setClockPart({ meridiem: value })
                }}
                disabled={isSubmitting}
              >
                <SelectTrigger className="w-full bg-secondary border-border text-foreground">
                  <SelectValue placeholder="AM/PM" />
                </SelectTrigger>
                <SelectContent className="bg-card border-border">
                  {MERIDIEM_OPTIONS.map((option) => (
                    <SelectItem key={option} value={option} className="text-foreground">
                      {option}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <p className="text-[11px] text-muted-foreground">Select hour, minute, and second for precise start timing.</p>
          </div>

          <div className="rounded-xl border border-border bg-secondary/40 p-3">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-foreground">Coverage Window</p>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  The upload starts at {startTime || "the selected start time"}. End time is computed from the processed video duration.
                </p>
              </div>
            </div>
          </div>

          <div className="rounded-xl border border-border bg-secondary/40 p-3">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                  <Video className="h-4 w-4 text-primary" />
                  Live Preview Window
                </div>
                <p className="mt-1 text-[11px] text-muted-foreground">
                  Show RT-DETR live frames in an OpenCV window while processing. Turn off to run fully in background.
                </p>
              </div>
              <Switch checked={showLivePreview} onCheckedChange={setShowLivePreview} disabled={isSubmitting} className="data-[state=checked]:bg-primary" />
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
            disabled={
              !selectedFile ||
              !locationId ||
              !date ||
              !startTime ||
              !selectedCountingConfig ||
              isSubmitting
            }
            className="bg-primary text-primary-foreground hover:bg-primary/90"
          >
            {isSubmitting ? "Adding video..." : "Add Video"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
