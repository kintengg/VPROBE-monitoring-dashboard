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

export interface LocationRecord {
  id: string
  name: string
  latitude: number
  longitude: number
  description: string
  address: string
  videos: VideoCard[]
}

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

export type TrafficPoint = { time: string } & Record<string, string | number>

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

export interface OcclusionTrendResponse {
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

export type OcclusionState = "clear" | "moderate" | "severe" | "no-footage" | "no-data"

export interface OcclusionHourScore {
  hour: string
  score: number
}

export interface OcclusionLocation {
  id: string
  name: string
  latitude: number
  longitude: number
  hasFootage: boolean
  hasOcclusionData: boolean
  score?: number | null
  state: OcclusionState
  hourlyScores: OcclusionHourScore[]
}

export interface OcclusionMapResponse {
  timeRange: string
  availableHours: string[]
  locations: OcclusionLocation[]
}

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
  thumbnailPath?: string | null
  previewPath?: string | null
  appearanceSummary?: string | null
  visualLabels?: string[]
  visualObjects?: string[]
  visualLogos?: string[]
  visualText?: string[]
  visualSummary?: string | null
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
  phase?: "queued" | "tracking" | "vision" | "finalizing" | null
  videoId?: string | null
  error?: string | null
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

export function createLocation(payload: Omit<LocationRecord, "id" | "videos">) {
  return request<LocationRecord>("/api/locations", {
    method: "POST",
    body: JSON.stringify(payload),
  })
}

export function updateLocation(locationId: string, payload: Omit<LocationRecord, "id" | "videos">) {
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
  return request<VideoRecord>(`/api/videos/${videoId}`)
}

export function getVideoUploadStatus(uploadId: string) {
  return request<VideoUploadStatus>(`/api/videos/uploads/${uploadId}`)
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
  return request<OcclusionMapResponse>(withQuery("/api/dashboard/occlusion", { date, timeRange }))
}

export function getDashboardOcclusionTrends(date?: string, timeRange = "whole-day", focusTime?: string, zoomLevel = 0) {
  return request<OcclusionTrendResponse>(withQuery("/api/dashboard/occlusion-trends", {
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
    filename: filenameMatch?.[1] ?? `dashboard-report-${date}-${timeRange}.md`,
  }
}