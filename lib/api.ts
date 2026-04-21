const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000"

export interface VideoCard {
  id: string
  timestamp: string
  date: string
  startTime: string
  endTime: string
  pedestrianCount: number
  rawPath?: string | null
  processedPath?: string | null
}

export type NormalizedPoint = [number, number]

export interface ROIConfiguration {
  referenceSize: [number, number]
  includePolygonsNorm: NormalizedPoint[][]
}

export type DirectionalCountLabel = "entering" | "exiting"

export type DirectionalStripKey = "strip_0" | "strip_1" | "strip_2"

export interface GateDirectionConfiguration {
  referenceSize: [number, number]
  gateDirectionZonesNorm: {
    strip_0: NormalizedPoint[]
    strip_1: NormalizedPoint[]
    strip_2: NormalizedPoint[]
  }
  directionMapping: {
    path_0_1_2: DirectionalCountLabel
    path_2_1_0: DirectionalCountLabel
  }
}

export interface VideoDirectionalEventRecord {
  trackId: string
  pedestrianId?: number | null
  direction: DirectionalCountLabel
  offsetSeconds: number
}

function isPositiveNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0
}

function parseDirectionalPoint(point: unknown): NormalizedPoint {
  if (!Array.isArray(point) || point.length !== 2) {
    throw new Error("Each point must be [x, y].")
  }

  const [x, y] = point
  if (
    typeof x !== "number"
    || typeof y !== "number"
    || !Number.isFinite(x)
    || !Number.isFinite(y)
    || x < 0
    || x > 1
    || y < 0
    || y > 1
  ) {
    throw new Error("Coordinates must be normalized between 0 and 1.")
  }

  return [x, y]
}

function parseDirectionalStrip(strip: unknown, stripName: DirectionalStripKey): NormalizedPoint[] {
  if (strip == null) {
    throw new Error(`${stripName} is required.`)
  }

  if (!Array.isArray(strip) || strip.length < 3) {
    throw new Error("Each strip must contain at least 3 points.")
  }

  return strip.map((point) => parseDirectionalPoint(point))
}

export function validateEntryExitPointsConfiguration(value: unknown): GateDirectionConfiguration {
  if (!value || typeof value !== "object") {
    throw new Error("Entry/Exit Points JSON must be valid JSON.")
  }

  const record = value as Record<string, unknown>
  const referenceSize = record.referenceSize
  if (
    !Array.isArray(referenceSize)
    || referenceSize.length !== 2
    || !isPositiveNumber(referenceSize[0])
    || !isPositiveNumber(referenceSize[1])
  ) {
    throw new Error("referenceSize must be [width, height].")
  }

  const gateDirectionZonesNorm = record.gateDirectionZonesNorm
  if (!gateDirectionZonesNorm || typeof gateDirectionZonesNorm !== "object") {
    throw new Error("gateDirectionZonesNorm must contain strip_0, strip_1, and strip_2.")
  }

  const zonesRecord = gateDirectionZonesNorm as Record<string, unknown>
  const strip_0 = parseDirectionalStrip(zonesRecord.strip_0, "strip_0")
  const strip_1 = parseDirectionalStrip(zonesRecord.strip_1, "strip_1")
  const strip_2 = parseDirectionalStrip(zonesRecord.strip_2, "strip_2")

  const directionMapping = record.directionMapping
  if (!directionMapping || typeof directionMapping !== "object") {
    throw new Error("directionMapping must define path_0_1_2 and path_2_1_0.")
  }

  const mappingRecord = directionMapping as Record<string, unknown>
  const path012 = mappingRecord.path_0_1_2
  const path210 = mappingRecord.path_2_1_0
  if (path012 == null || path210 == null) {
    throw new Error("directionMapping must define path_0_1_2 and path_2_1_0.")
  }

  if (
    (path012 !== "entering" && path012 !== "exiting")
    || (path210 !== "entering" && path210 !== "exiting")
  ) {
    throw new Error("directionMapping values must be either entering or exiting.")
  }

  if (path012 === path210) {
    throw new Error("directionMapping values must map one path to entering and the other to exiting.")
  }

  return {
    referenceSize: [referenceSize[0], referenceSize[1]],
    gateDirectionZonesNorm: {
      strip_0,
      strip_1,
      strip_2,
    },
    directionMapping: {
      path_0_1_2: path012,
      path_2_1_0: path210,
    },
  }
}

export function parseEntryExitPointsConfiguration(value: string): GateDirectionConfiguration | null {
  const trimmed = value.trim()
  if (!trimmed) {
    return null
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch {
    throw new Error("Entry/Exit Points JSON must be valid JSON.")
  }

  return validateEntryExitPointsConfiguration(parsed)
}

export function hasValidEntryExitPointsConfiguration(value: unknown): value is GateDirectionConfiguration {
  try {
    validateEntryExitPointsConfiguration(value)
    return true
  } catch {
    return false
  }
}

export interface LocationRecord {
  id: string
  name: string
  latitude: number
  longitude: number
  description: string
  address: string
  roiCoordinates?: ROIConfiguration | null
  entryExitPoints?: GateDirectionConfiguration | null
  walkableAreaM2?: number | null
  videos: VideoCard[]
}

export type LocationPayload = Omit<LocationRecord, "id" | "videos">

export interface LocationSearchResult {
  name: string
  address: string
  latitude: number
  longitude: number
  placeId?: string | null
  types: string[]
}

export interface VideoRecord {
  id: string
  locationId: string
  location: string
  timestamp: string
  date: string
  startTime: string
  endTime: string
  gpsLat: number
  gpsLng: number
  pedestrianCount: number
  rawPath?: string | null
  processedPath?: string | null
  severitySummary?: VideoSeveritySummary | null
}

export interface VideoPedestrianTrackRecord {
  id: string
  pedestrianId?: number | null
  firstOffsetSeconds: number
  lastOffsetSeconds: number
}

export interface VideoDetailRecord extends VideoRecord {
  pedestrianTracks: VideoPedestrianTrackRecord[]
  directionalEvents: VideoDirectionalEventRecord[]
}

export type VideoSeverityLevel = "neutral" | "light" | "moderate" | "heavy"

export interface VideoSeverityBucket {
  startOffsetSeconds: number
  endOffsetSeconds: number
  severity: VideoSeverityLevel
  score?: number | null
}

export interface VideoSeveritySummary {
  bucketCount: number
  sampledSeconds: number
  buckets: VideoSeverityBucket[]
}

export interface EventRecord {
  id: string
  type: "detection" | "alert" | "motion"
  location: string
  timestamp: string
  description: string
  videoId?: string | null
  pedestrianId?: number | null
  frame?: number | null
  offsetSeconds?: number | null
  occlusionClass?: number | null
}

export interface DashboardSummary {
  totalUniquePedestrians: number
  averageFps: number
  totalHeavyOcclusions: number
  monitoredLocations: number
}

export interface LocationTotal {
  location: string
  totalPedestrians: number
}

export type TrafficPoint = { id: string; time: string } & Record<string, string | number>

export interface TrafficResponse {
  timeRange: string
  series: TrafficPoint[]
  bucketMinutes: number
  zoomLevel: number
  canZoomIn: boolean
  isDrilldown: boolean
  focusTime?: string | null
  windowStart?: string | null
  windowEnd?: string | null
  locationTotals: LocationTotal[]
}

export interface PTSITrendResponse {
  timeRange: string
  series: TrafficPoint[]
  bucketMinutes: number
  zoomLevel: number
  canZoomIn: boolean
  isDrilldown: boolean
  focusTime?: string | null
  windowStart?: string | null
  windowEnd?: string | null
}

export type PTSIState = "clear" | "moderate" | "severe" | "no-footage" | "no-data"
export type PTSILOS = "A" | "B" | "C" | "D" | "E" | "F"

export type PTSIMode = "strict-fhwa" | "roi-testing"

export interface PTSIOcclusionMix {
  lightPercent: number
  moderatePercent: number
  heavyPercent: number
}

export interface PTSIHourScore {
  hour: string
  score: number
  mode?: PTSIMode | null
  averagePedestrians?: number | null
  uniquePedestrians?: number | null
  occlusionMix?: PTSIOcclusionMix | null
  los?: PTSILOS | null
  losDescription?: string | null
}

export interface PTSILocation {
  id: string
  name: string
  latitude: number
  longitude: number
  hasFootage: boolean
  hasPTSIData: boolean
  score?: number | null
  state: PTSIState
  mode?: PTSIMode | null
  averagePedestrians?: number | null
  uniquePedestrians?: number | null
  occlusionMix?: PTSIOcclusionMix | null
  los?: PTSILOS | null
  losDescription?: string | null
  peakHour?: string | null
  peakHourScore?: number | null
  offPeakHour?: string | null
  offPeakHourScore?: number | null
  hourlyScores: PTSIHourScore[]
}

export interface PTSIMapResponse {
  timeRange: string
  availableHours: string[]
  locations: PTSILocation[]
}

export type OcclusionTrendResponse = PTSITrendResponse
export type OcclusionState = PTSIState
export type OcclusionHourScore = PTSIHourScore
export type OcclusionLocation = PTSILocation
export type OcclusionMapResponse = PTSIMapResponse

export interface AIBadge {
  label: string
  value: string
  tone: "blue" | "green" | "orange" | "purple" | "red"
}

export interface AISection {
  title: string
  body: string
  badges: AIBadge[]
}

export interface AISynthesisResponse {
  date: string
  timeRange: string
  sections: AISection[]
}

export interface SearchResult {
  id: string
  videoId: string
  timestamp: string
  date: string
  location: string
  confidence: number
  matchReason: string
  pedestrianId?: number | null
  frame?: number | null
  offsetSeconds?: number | null
  firstTimestamp?: string | null
  lastTimestamp?: string | null
  firstOffsetSeconds?: number | null
  lastOffsetSeconds?: number | null
  thumbnailPath?: string | null
  previewPath?: string | null
  appearanceSummary?: string | null
  visualLabels?: string[]
  visualObjects?: string[]
  visualLogos?: string[]
  visualText?: string[]
  visualSummary?: string | null
  semanticScore?: number | null
  possibleMatch?: boolean
  matchStrategy?: "semantic" | "metadata" | "event" | null
}

export interface ModelInfo {
  currentModel?: string | null
  uploadedAt?: string | null
}

export interface DownloadedReport {
  blob: Blob
  filename: string
}

export interface VideoUploadStatus {
  uploadId: string
  state: "queued" | "processing" | "complete" | "error" | "cancelled"
  progressPercent?: number | null
  message: string
  phase?: "queued" | "tracking" | "vision" | "ptsi" | "finalizing" | null
  videoId?: string | null
  error?: string | null
  fileName?: string | null
  locationId?: string | null
  locationName?: string | null
  date?: string | null
  startTime?: string | null
  endTime?: string | null
  fastMode?: boolean | null
  createdAt?: string | null
  startedAt?: string | null
  completedAt?: string | null
  updatedAt: string
}

export interface MediaPathSource {
  rawPath?: string | null
  processedPath?: string | null
}

function createUploadId() {
  if (typeof globalThis.crypto !== "undefined" && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID()
  }

  return `upload-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function isTerminalUploadState(state: VideoUploadStatus["state"]) {
  return state === "complete" || state === "error" || state === "cancelled"
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const isFormData = typeof FormData !== "undefined" && init?.body instanceof FormData
  const response = await fetch(`${API_BASE_URL}${path}`, {
    cache: "no-store",
    ...init,
    headers: {
      ...(isFormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
  })

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`

    try {
      const payload = await response.json()
      if (typeof payload?.detail === "string") {
        message = payload.detail
      }
    } catch {
      // Ignore JSON parsing issues and keep the fallback message.
    }

    throw new Error(message)
  }

  if (response.status === 204) {
    return undefined as T
  }

  return response.json() as Promise<T>
}

async function parseError(response: Response) {
  let message = `Request failed with status ${response.status}`

  try {
    const payload = await response.json()
    if (typeof payload?.detail === "string") {
      message = payload.detail
    }
  } catch {
    // Ignore JSON parsing issues and keep the fallback message.
  }

  return message
}

function withQuery(path: string, params: Record<string, string | undefined>) {
  const search = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value) search.set(key, value)
  })
  const query = search.toString()
  return query ? `${path}?${query}` : path
}

export function getMediaUrl(path?: string | null) {
  if (!path) return null
  if (/^https?:\/\//i.test(path)) return path

  const normalizedPath = path.startsWith("/") ? path : `/${path}`
  return `${API_BASE_URL}${normalizedPath}`
}

export function getVideoPlaybackPath(video: MediaPathSource, preferProcessed = true) {
  if (preferProcessed) {
    return video.processedPath ?? video.rawPath ?? null
  }

  return video.rawPath ?? video.processedPath ?? null
}

export function getLocations(date?: string) {
  return request<LocationRecord[]>(withQuery("/api/locations", { date }))
}

export function searchLocations(query: string) {
  return request<LocationSearchResult[]>(withQuery("/api/locations/search", { query }))
}

export function createLocation(payload: LocationPayload) {
  return request<LocationRecord>("/api/locations", {
    method: "POST",
    body: JSON.stringify(payload),
  })
}

export function updateLocation(locationId: string, payload: LocationPayload) {
  return request<LocationRecord>(`/api/locations/${locationId}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  })
}

export function deleteLocation(locationId: string) {
  return request<void>(`/api/locations/${locationId}`, {
    method: "DELETE",
  })
}

export function getVideo(videoId: string) {
  return request<VideoDetailRecord>(`/api/videos/${videoId}`)
}

export function getVideoUploadStatus(uploadId: string) {
  return request<VideoUploadStatus>(`/api/videos/uploads/${uploadId}`)
}

export function getVideoUploadHistory() {
  return request<VideoUploadStatus[]>("/api/videos/uploads/history")
}

export function cancelVideoUpload(uploadId: string) {
  return request<VideoUploadStatus>(`/api/videos/uploads/${uploadId}/cancel`, {
    method: "POST",
  })
}

export function uploadVideo(payload: {
  file: File
  locationId: string
  date: string
  startTime: string
  endTime: string
  fastMode?: boolean
  onProgress?: (status: VideoUploadStatus) => void
}) {
  const formData = new FormData()
  const uploadId = createUploadId()
  formData.set("file", payload.file)
  formData.set("locationId", payload.locationId)
  formData.set("date", payload.date)
  formData.set("startTime", payload.startTime)
  formData.set("endTime", payload.endTime)
  formData.set("fastMode", String(Boolean(payload.fastMode)))
  formData.set("uploadId", uploadId)

  payload.onProgress?.({
    uploadId,
    state: "queued",
    progressPercent: 0,
    message: payload.fastMode ? "Uploading video in fast mode..." : "Uploading video...",
    phase: "queued",
    updatedAt: new Date().toISOString(),
  })

  let shouldPoll = Boolean(payload.onProgress)

  const pollUploadStatus = async () => {
    if (!payload.onProgress) {
      return null
    }

    let lastStatus: VideoUploadStatus | null = null

    while (shouldPoll) {
      try {
        const status = await getVideoUploadStatus(uploadId)
        lastStatus = status
        payload.onProgress(status)

        if (isTerminalUploadState(status.state)) {
          shouldPoll = false
          return status
        }
      } catch {
        if (!shouldPoll) {
          break
        }
      }

      if (!shouldPoll) {
        break
      }

      await sleep(500)
    }

    return lastStatus
  }

  const pollingPromise = pollUploadStatus()

  return (async () => {
    try {
      const result = await request<VideoRecord>("/api/videos", {
        method: "POST",
        body: formData,
      })

      shouldPoll = false

      if (payload.onProgress) {
        try {
          payload.onProgress(await getVideoUploadStatus(uploadId))
        } catch {
          payload.onProgress({
            uploadId,
            state: "complete",
            progressPercent: 100,
            message: "Video upload and processing complete.",
            videoId: result.id,
            updatedAt: new Date().toISOString(),
          })
        }
      }

      return result
    } catch (error) {
      shouldPoll = false

      if (payload.onProgress) {
        try {
          payload.onProgress(await getVideoUploadStatus(uploadId))
        } catch {
          // Keep the original request error when status lookup is unavailable.
        }
      }

      throw error
    } finally {
      await pollingPromise
    }
  })()
}

export function deleteVideo(videoId: string) {
  return request<void>(`/api/videos/${videoId}`, {
    method: "DELETE",
  })
}

export function getEvents(videoId?: string) {
  return request<EventRecord[]>(withQuery("/api/events", { videoId }))
}

export function getDashboardSummary(date?: string) {
  return request<DashboardSummary>(withQuery("/api/dashboard/summary", { date }))
}

export function getDashboardTraffic(date?: string, timeRange = "whole-day", focusTime?: string, zoomLevel = 0) {
  return request<TrafficResponse>(withQuery("/api/dashboard/traffic", {
    date,
    timeRange,
    focusTime,
    zoomLevel: zoomLevel > 0 ? String(zoomLevel) : undefined,
  }))
}

export function getDashboardOcclusion(date?: string, timeRange = "whole-day") {
  return request<PTSIMapResponse>(withQuery("/api/dashboard/occlusion", { date, timeRange }))
}

export function getDashboardOcclusionTrends(date?: string, timeRange = "whole-day", focusTime?: string, zoomLevel = 0) {
  return request<PTSITrendResponse>(withQuery("/api/dashboard/occlusion-trends", {
    date,
    timeRange,
    focusTime,
    zoomLevel: zoomLevel > 0 ? String(zoomLevel) : undefined,
  }))
}

export function getAISynthesis(date: string, timeRange: string) {
  return request<AISynthesisResponse>(withQuery("/api/dashboard/ai-synthesis", { date, timeRange }))
}

export function searchVideos(query: string) {
  return request<SearchResult[]>(withQuery("/api/search", { query }))
}

export function getCurrentModel() {
  return request<ModelInfo>("/api/models/current")
}

export function uploadModel(file: File) {
  const formData = new FormData()
  formData.set("file", file)

  return request<ModelInfo>("/api/models/upload", {
    method: "POST",
    body: formData,
  })
}

export async function downloadDashboardReport(date: string, timeRange: string): Promise<DownloadedReport> {
  const response = await fetch(`${API_BASE_URL}${withQuery("/api/dashboard/export", { date, timeRange })}`, {
    cache: "no-store",
  })

  if (!response.ok) {
    throw new Error(await parseError(response))
  }

  const blob = await response.blob()
  const contentDisposition = response.headers.get("content-disposition")
  const filenameMatch = contentDisposition?.match(/filename="?([^";]+)"?/) ?? null

  return {
    blob,
    filename: filenameMatch?.[1] ?? `${date}-${timeRange}-dashboard-report.zip`,
  }
}