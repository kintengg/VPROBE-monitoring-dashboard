"use client"

import { Suspense, use, useEffect, useMemo, useRef, useState } from "react"
import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation"
import { VideoPlayer } from "@/components/video/video-player"
import { VideoMetadata } from "@/components/video/video-metadata"
import { PlaybackTimeline } from "@/components/video/playback-timeline"
import { EventFeed } from "@/components/surveillance/event-feed"
import { AISearchBar } from "@/components/surveillance/ai-search-bar"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { AlertCircle, ArrowLeft, Download, Footprints, Loader2, LogIn, LogOut, Share2, Trash2, Users } from "lucide-react"
import {
  deleteVideo,
  getEvents,
  getLocations,
  getMediaUrl,
  getVideo,
  getVideoPlaybackPath,
  hasValidEntryExitPointsConfiguration,
  type EventRecord,
  type LocationRecord,
  type VideoDetailRecord,
  type VideoPedestrianTrackRecord,
} from "@/lib/api"

function getDetectionStatus(event: EventRecord) {
  if (event.type === "alert") return "Requires Review"
  if (event.type === "motion") return "Motion Event"
  return "Tracked"
}

function createPlaybackWindow(start: number, end: number) {
  const safeStart = Math.max(0, start)
  const safeEnd = Math.max(safeStart, end)
  return {
    start: safeStart,
    end: safeEnd > safeStart ? safeEnd : safeStart + 0.5,
  }
}

function trackPlaybackWindows(tracks: VideoPedestrianTrackRecord[]) {
  return tracks
    .filter(
      (track) =>
        Number.isFinite(track.firstOffsetSeconds)
        && Number.isFinite(track.lastOffsetSeconds),
    )
    .map((track) => createPlaybackWindow(track.firstOffsetSeconds, track.lastOffsetSeconds))
}

function parseClockSeconds(value?: string | null) {
  if (!value) return null

  const parts = value.split(":").map((part) => Number(part))
  if (parts.length < 2 || parts.length > 3 || parts.some((part) => !Number.isFinite(part) || part < 0)) {
    return null
  }

  const [hours, minutes, seconds = 0] = parts
  return (hours * 3600) + (minutes * 60) + seconds
}

function scheduledDurationSeconds(startTime?: string | null, endTime?: string | null) {
  const startSeconds = parseClockSeconds(startTime)
  const endSeconds = parseClockSeconds(endTime)
  if (startSeconds === null || endSeconds === null) {
    return null
  }

  const normalizedEndSeconds = endSeconds >= startSeconds ? endSeconds : endSeconds + (24 * 3600)
  const durationSeconds = normalizedEndSeconds - startSeconds
  return durationSeconds > 0 ? durationSeconds : null
}

function sourceTimeFromPlaybackTime(playbackSeconds: number, playbackDurationSeconds: number, sourceDurationSeconds: number) {
  const safePlaybackSeconds = Math.max(0, playbackSeconds)
  if (!(playbackDurationSeconds > 0) || !(sourceDurationSeconds > 0)) {
    return safePlaybackSeconds
  }

  if (Math.abs(playbackDurationSeconds - sourceDurationSeconds) < 0.01) {
    return Math.min(safePlaybackSeconds, sourceDurationSeconds)
  }

  return Math.min(sourceDurationSeconds, safePlaybackSeconds * (sourceDurationSeconds / playbackDurationSeconds))
}

function playbackTimeFromSourceTime(sourceSeconds: number, playbackDurationSeconds: number, sourceDurationSeconds: number) {
  const safeSourceSeconds = Math.max(0, sourceSeconds)
  if (!(playbackDurationSeconds > 0) || !(sourceDurationSeconds > 0)) {
    return safeSourceSeconds
  }

  if (Math.abs(playbackDurationSeconds - sourceDurationSeconds) < 0.01) {
    return Math.min(safeSourceSeconds, playbackDurationSeconds)
  }

  return Math.min(playbackDurationSeconds, safeSourceSeconds * (playbackDurationSeconds / sourceDurationSeconds))
}

const LIVE_DETECTION_EPSILON_SECONDS = 0.25

function VideoDetailContent({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params)
  const router = useRouter()
  const searchParams = useSearchParams()
  const [video, setVideo] = useState<VideoDetailRecord | null>(null)
  const [videoLocation, setVideoLocation] = useState<LocationRecord | null>(null)
  const [events, setEvents] = useState<EventRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [selectedEventId, setSelectedEventId] = useState<string | undefined>(undefined)
  const [requestedSeek, setRequestedSeek] = useState<{ seconds: number; token: number } | null>(null)
  const [requestedSeekSourceSeconds, setRequestedSeekSourceSeconds] = useState<number | null>(null)
  const [playbackTimeSeconds, setPlaybackTimeSeconds] = useState(0)
  const [playbackDurationSeconds, setPlaybackDurationSeconds] = useState(0)
  const [showAllDetections, setShowAllDetections] = useState(false)
  const [showROI, setShowROI] = useState(false)
  const [showEntryExitPoints, setShowEntryExitPoints] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const videoElementRef = useRef<HTMLVideoElement | null>(null)
  const seekTokenRef = useRef(0)
  const appliedQuerySeekRef = useRef("")

  useEffect(() => {
    let cancelled = false

    const loadVideo = async () => {
      setLoading(true)
      try {
        const [videoResponse, eventsResponse, locationsResponse] = await Promise.all([
          getVideo(id),
          getEvents(id),
          getLocations().catch(() => null),
        ])

        if (!cancelled) {
          setVideo(videoResponse)
          setVideoLocation((locationsResponse ?? []).find((location) => location.id === videoResponse.locationId) ?? null)
          setEvents(eventsResponse)
          setError(null)
          setActionError(null)
        }
      } catch (error) {
        if (!cancelled) {
          setError(error instanceof Error ? error.message : "Failed to load video details.")
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void loadVideo()

    return () => {
      cancelled = true
    }
  }, [id])

  useEffect(() => {
    setShowAllDetections(false)
    setShowROI(false)
    setShowEntryExitPoints(false)
    setRequestedSeek(null)
    setRequestedSeekSourceSeconds(null)
    setPlaybackTimeSeconds(0)
    setPlaybackDurationSeconds(0)
  }, [id])

  const orderedEvents = useMemo(
    () =>
      [...events].sort((left, right) => {
        const leftOffset = typeof left.offsetSeconds === "number" ? left.offsetSeconds : Number.POSITIVE_INFINITY
        const rightOffset = typeof right.offsetSeconds === "number" ? right.offsetSeconds : Number.POSITIVE_INFINITY
        return leftOffset - rightOffset
      }),
    [events],
  )

  const sourceDurationSeconds = useMemo(() => {
    const candidates: number[] = []
    const scheduledDuration = scheduledDurationSeconds(video?.startTime, video?.endTime)

    if (scheduledDuration !== null) {
      candidates.push(scheduledDuration)
    }

    for (const track of video?.pedestrianTracks ?? []) {
      if (Number.isFinite(track.lastOffsetSeconds)) {
        candidates.push(track.lastOffsetSeconds)
      }
    }

    for (const directionalEvent of video?.directionalEvents ?? []) {
      if (Number.isFinite(directionalEvent.offsetSeconds)) {
        candidates.push(directionalEvent.offsetSeconds)
      }
    }

    for (const event of orderedEvents) {
      if (typeof event.offsetSeconds === "number" && Number.isFinite(event.offsetSeconds)) {
        candidates.push(event.offsetSeconds)
      }
    }

    for (const bucket of video?.severitySummary?.buckets ?? []) {
      if (Number.isFinite(bucket.endOffsetSeconds)) {
        candidates.push(bucket.endOffsetSeconds)
      }
    }

    return candidates.length > 0 ? Math.max(...candidates) : playbackDurationSeconds
  }, [orderedEvents, playbackDurationSeconds, video?.directionalEvents, video?.endTime, video?.pedestrianTracks, video?.severitySummary?.buckets, video?.startTime])

  const currentTimeSeconds = useMemo(
    () => sourceTimeFromPlaybackTime(playbackTimeSeconds, playbackDurationSeconds, sourceDurationSeconds),
    [playbackDurationSeconds, playbackTimeSeconds, sourceDurationSeconds],
  )

  useEffect(() => {
    const eventId = searchParams.get("eventId") ?? undefined
    setSelectedEventId(eventId)

    const seekValue = searchParams.get("seek")
    const seekKey = `${id}:${eventId ?? ""}:${seekValue ?? ""}`

    if (!seekValue || appliedQuerySeekRef.current === seekKey) {
      return
    }

    const seconds = Number(seekValue)
    if (!Number.isFinite(seconds)) {
      return
    }

    appliedQuerySeekRef.current = seekKey
    setRequestedSeekSourceSeconds(seconds)
    setPlaybackTimeSeconds(playbackTimeFromSourceTime(seconds, playbackDurationSeconds, sourceDurationSeconds))
  }, [id, playbackDurationSeconds, searchParams, sourceDurationSeconds])

  useEffect(() => {
    if (requestedSeekSourceSeconds === null) {
      return
    }

    const playbackSeekSeconds = playbackTimeFromSourceTime(
      requestedSeekSourceSeconds,
      playbackDurationSeconds,
      sourceDurationSeconds,
    )

    seekTokenRef.current += 1
    setRequestedSeek({ seconds: playbackSeekSeconds, token: seekTokenRef.current })
  }, [playbackDurationSeconds, requestedSeekSourceSeconds, sourceDurationSeconds])

  const searchMatchOffsets = useMemo(() => {
    const rawMatches = searchParams.get("matches")
    const parsedMatches = (rawMatches ?? "")
      .split(",")
      .map((value) => Number(value.trim()))
      .filter((value) => Number.isFinite(value) && value >= 0)

    if (parsedMatches.length > 0) {
      return Array.from(new Set(parsedMatches)).sort((left, right) => left - right)
    }

    const seekValue = Number(searchParams.get("seek"))
    return Number.isFinite(seekValue) && seekValue >= 0 ? [seekValue] : []
  }, [searchParams])

  const detectionDetails = useMemo(() => {
    const seen = new Set<number>()

    return orderedEvents
      .filter((event): event is EventRecord & { pedestrianId: number } => typeof event.pedestrianId === "number")
      .filter((event) => {
        if (seen.has(event.pedestrianId)) return false
        seen.add(event.pedestrianId)
        return true
      })
      .map((event) => ({ id: event.pedestrianId, status: getDetectionStatus(event) }))
  }, [orderedEvents])

  const pedestrianPlaybackWindows = useMemo(() => {
    const trackWindows = trackPlaybackWindows(video?.pedestrianTracks ?? [])
    if (trackWindows.length > 0) {
      return trackWindows
    }

    const windows = new Map<number, { start: number; end: number }>()

    for (const event of orderedEvents) {
      if (event.type !== "detection" || typeof event.pedestrianId !== "number" || typeof event.offsetSeconds !== "number") {
        continue
      }

      const offset = Math.max(0, event.offsetSeconds)
      const existingWindow = windows.get(event.pedestrianId)

      if (!existingWindow) {
        windows.set(event.pedestrianId, { start: offset, end: offset })
        continue
      }

      existingWindow.start = Math.min(existingWindow.start, offset)
      existingWindow.end = Math.max(existingWindow.end, offset)
    }

    return Array.from(windows.values()).map((window) => createPlaybackWindow(window.start, window.end))
  }, [orderedEvents, video?.pedestrianTracks])

  const trackedPedestriansSoFar = useMemo(() => {
    return pedestrianPlaybackWindows.reduce((count, window) => (window.start <= currentTimeSeconds ? count + 1 : count), 0)
  }, [currentTimeSeconds, pedestrianPlaybackWindows])

  const liveDetectedCount = useMemo(
    () =>
      pedestrianPlaybackWindows.reduce(
        (count, window) => (
          currentTimeSeconds >= window.start
          && currentTimeSeconds <= (window.end + LIVE_DETECTION_EPSILON_SECONDS)
            ? count + 1
            : count
        ),
        0,
      ),
    [currentTimeSeconds, pedestrianPlaybackWindows],
  )

  const visibleDetectionDetails = showAllDetections ? detectionDetails : detectionDetails.slice(0, 15)
  const hasCollapsedDetections = detectionDetails.length > 15
  const hasLocationROI = Boolean(videoLocation?.roiCoordinates?.includePolygonsNorm?.length)
  const hasEntryExitPoints = hasValidEntryExitPointsConfiguration(videoLocation?.entryExitPoints)
  const directionalEvents = useMemo(
    () => (hasEntryExitPoints ? [...(video?.directionalEvents ?? [])].sort((left, right) => left.offsetSeconds - right.offsetSeconds) : []),
    [hasEntryExitPoints, video?.directionalEvents],
  )
  const enteringCount = useMemo(
    () => directionalEvents.reduce((count, event) => (event.direction === "entering" && event.offsetSeconds <= currentTimeSeconds ? count + 1 : count), 0),
    [currentTimeSeconds, directionalEvents],
  )
  const exitingCount = useMemo(
    () => directionalEvents.reduce((count, event) => (event.direction === "exiting" && event.offsetSeconds <= currentTimeSeconds ? count + 1 : count), 0),
    [currentTimeSeconds, directionalEvents],
  )

  const mediaUrl = video ? getMediaUrl(getVideoPlaybackPath(video)) : null
  const metricTiles = [
    {
      icon: Users,
      value: String(trackedPedestriansSoFar),
      caption: "Total so far",
      iconClassName: "bg-emerald-500 text-white ring-emerald-500/20",
      valueClassName: "text-emerald-100 sm:text-foreground",
      cardClassName: "border-emerald-400/30 bg-gradient-to-br from-emerald-500/18 via-green-500/10 to-transparent",
    },
    {
      icon: Footprints,
      value: String(liveDetectedCount),
      caption: "Detected now",
      iconClassName: "bg-cyan-500 text-white ring-cyan-500/20",
      valueClassName: "text-cyan-100 sm:text-foreground",
      cardClassName: "border-cyan-400/30 bg-gradient-to-br from-cyan-500/20 via-sky-500/10 to-transparent",
    },
    {
      icon: LogIn,
      value: hasEntryExitPoints ? String(enteringCount) : "--",
      caption: "Entering so far",
      iconClassName: "bg-violet-500/90 text-white ring-violet-500/20",
      valueClassName: "text-foreground",
      cardClassName: "border-violet-400/25 bg-gradient-to-br from-violet-500/12 via-fuchsia-500/8 to-transparent",
    },
    {
      icon: LogOut,
      value: hasEntryExitPoints ? String(exitingCount) : "--",
      caption: "Exiting so far",
      iconClassName: "bg-amber-500/90 text-white ring-amber-500/20",
      valueClassName: "text-foreground",
      cardClassName: "border-amber-400/25 bg-gradient-to-br from-amber-500/12 via-orange-500/8 to-transparent",
    },
  ]

  const requestSeek = (seconds: number) => {
    const safeSeconds = Math.max(0, seconds)
    setRequestedSeekSourceSeconds(safeSeconds)
    setPlaybackTimeSeconds(playbackTimeFromSourceTime(safeSeconds, playbackDurationSeconds, sourceDurationSeconds))
  }

  const handleEventSelect = (event: EventRecord) => {
    setSelectedEventId(event.id)
    setActionError(null)

    if (typeof event.offsetSeconds === "number") {
      requestSeek(event.offsetSeconds)
    }

    const params = new URLSearchParams(searchParams.toString())
    params.set("eventId", event.id)
    if (typeof event.offsetSeconds === "number") {
      params.set("seek", String(event.offsetSeconds))
    } else {
      params.delete("seek")
    }

    const query = params.toString()
    router.replace(query ? `/video/${id}?${query}` : `/video/${id}`, { scroll: false })
  }

  const handleDelete = async () => {
    if (deleting || !video) return

    const confirmed = typeof window === "undefined"
      ? false
      : window.confirm(`Delete the recording for ${video.location} on ${video.date}? This also removes the saved media files.`)

    if (!confirmed) return

    setDeleting(true)
    setActionError(null)

    try {
      await deleteVideo(video.id)
      router.push("/")
      router.refresh()
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Failed to delete this video.")
      setDeleting(false)
    }
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground">
        <Loader2 className="mr-2 h-6 w-6 animate-spin" />
        Loading video details...
      </div>
    )
  }

  if (error || !video) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <div className="max-w-md rounded-3xl border border-destructive/30 bg-card p-6 text-center shadow-elevated-sm">
          <AlertCircle className="mx-auto mb-4 h-10 w-10 text-destructive" />
          <h1 className="text-xl font-semibold text-foreground">Unable to load this video</h1>
          <p className="mt-2 text-sm text-muted-foreground">{error ?? "The requested video could not be found."}</p>
          <Link href="/" className="mt-4 inline-block">
            <Button variant="outline" className="border-border text-foreground hover:bg-secondary">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to overview
            </Button>
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full">
      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-border bg-card">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
                <ArrowLeft className="w-5 h-5" />
              </Button>
            </Link>
            <div>
              <h1 className="text-xl font-semibold text-foreground">{video.location}</h1>
              <p className="text-sm text-muted-foreground">Video Feed #{id}</p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <Button
              variant="destructive"
              className="rounded-2xl"
              onClick={() => void handleDelete()}
              disabled={deleting}
            >
              {deleting ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Trash2 className="w-4 h-4 mr-2" />}
              Delete
            </Button>
            <Button
              variant="outline"
              className="border-border text-foreground hover:bg-secondary rounded-2xl"
              onClick={() => {
                if (typeof window !== "undefined") {
                  void navigator.clipboard.writeText(window.location.href)
                }
              }}
            >
              <Share2 className="w-4 h-4 mr-2" />
              Copy Link
            </Button>
            {mediaUrl ? (
              <Button asChild className="bg-primary text-primary-foreground hover:bg-primary/90 rounded-2xl">
                <a href={mediaUrl} download>
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </a>
              </Button>
            ) : (
              <Button className="bg-primary text-primary-foreground hover:bg-primary/90 rounded-2xl" disabled>
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            )}
          </div>
        </header>

        {/* Video Player and Controls */}
        <div className="flex-1 overflow-auto p-6">
          <div className="max-w-5xl mx-auto space-y-6">
            {actionError && (
              <div className="flex items-start gap-3 rounded-2xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                <span>{actionError}</span>
              </div>
            )}

            {/* Video Player with Bounding Boxes */}
            <VideoPlayer
              videoId={video.id}
              location={video.location}
              src={mediaUrl}
              pedestrianCount={video.pedestrianCount}
              timestamp={video.timestamp}
              date={video.date}
              isProcessed={Boolean(video.processedPath)}
              videoRef={videoElementRef}
              requestedSeek={requestedSeek}
              roiCoordinates={videoLocation?.roiCoordinates ?? null}
              showROI={showROI}
              entryExitPoints={hasEntryExitPoints ? (videoLocation?.entryExitPoints ?? null) : null}
              showEntryExitPoints={showEntryExitPoints}
              onTimeUpdate={setPlaybackTimeSeconds}
              onDurationChange={setPlaybackDurationSeconds}
            />

            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              {metricTiles.map((tile) => {
                const Icon = tile.icon
                return (
                  <div
                    key={tile.caption}
                    className={`flex items-center gap-3 rounded-2xl border px-4 py-3 shadow-elevated-sm ${tile.cardClassName}`}
                  >
                    <div className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl ring-4 ${tile.iconClassName}`}>
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="min-w-0">
                      <p className={`text-2xl font-semibold leading-none ${tile.valueClassName}`}>{tile.value}</p>
                      <p className="mt-1 text-xs text-muted-foreground">{tile.caption}</p>
                    </div>
                  </div>
                )
              })}
            </div>
            
            {/* Playback Timeline */}
            <PlaybackTimeline
              startTime={video.startTime}
              endTime={video.endTime}
              durationSeconds={sourceDurationSeconds}
              currentTimeSeconds={currentTimeSeconds}
              events={orderedEvents}
              severityBuckets={video.severitySummary?.buckets ?? []}
              searchMatchOffsets={searchMatchOffsets}
              onSeek={requestSeek}
            />
            
            {/* Metadata Section */}
            <VideoMetadata 
              date={video.date}
              startTime={video.startTime}
              endTime={video.endTime}
              gpsLat={video.gpsLat}
              gpsLng={video.gpsLng}
              trackedPedestriansSoFar={trackedPedestriansSoFar}
              pedestrianCount={video.pedestrianCount}
            />

            {(hasLocationROI || hasEntryExitPoints) && (
              <div className="space-y-3">
                {hasLocationROI && (
                  <div className="flex items-center justify-between gap-4 rounded-2xl border border-border/70 bg-card/70 px-4 py-3 shadow-elevated-sm">
                    <div>
                      <p className="text-sm font-medium text-foreground">Show ROI Outline</p>
                      <p className="text-xs text-muted-foreground">Display the stored walkable ROI polygons over the video for alignment debugging.</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-medium text-muted-foreground">{showROI ? "ON" : "OFF"}</span>
                      <Switch checked={showROI} onCheckedChange={setShowROI} aria-label="Show ROI Outline" />
                    </div>
                  </div>
                )}
                {hasEntryExitPoints && (
                  <div className="flex items-center justify-between gap-4 rounded-2xl border border-border/70 bg-card/70 px-4 py-3 shadow-elevated-sm">
                    <div>
                      <p className="text-sm font-medium text-foreground">Show ROI Outline (Entry/Exit Points)</p>
                      <p className="text-xs text-muted-foreground">Display the stored directional gate strips over the video for alignment and counting debugging.</p>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs font-medium text-muted-foreground">{showEntryExitPoints ? "ON" : "OFF"}</span>
                      <Switch checked={showEntryExitPoints} onCheckedChange={setShowEntryExitPoints} aria-label="Show ROI Outline (Entry/Exit Points)" />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right Sidebar - Filtered Event Feed */}
      <aside className="w-80 border-l border-border bg-card flex flex-col h-full">
        <AISearchBar />
        <EventFeed
          filteredVideoId={id}
          events={orderedEvents}
          loading={loading}
          selectedEventId={selectedEventId}
          onEventSelect={handleEventSelect}
        />
        
        {/* Detection Details */}
        <div className="border-t border-border p-4">
          <h4 className="text-sm font-medium text-foreground mb-3">Detection Details</h4>
          {detectionDetails.length > 0 ? (
            <div className="space-y-2">
              {visibleDetectionDetails.map((detail) => (
                <DetectionDetail key={detail.id} id={detail.id} status={detail.status} />
              ))}
              {hasCollapsedDetections && (
                <Button
                  variant="outline"
                  className="w-full rounded-2xl border-border text-foreground hover:bg-secondary"
                  onClick={() => setShowAllDetections((current) => !current)}
                >
                  {showAllDetections ? "Show less" : `View ${detectionDetails.length - 15} more`}
                </Button>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No tracked pedestrian IDs are available for this video yet.</p>
          )}
        </div>
      </aside>
    </div>
  )
}

export default function VideoDetailPage({ params }: { params: Promise<{ id: string }> }) {
  return (
    <Suspense fallback={
      <div className="flex h-full items-center justify-center text-muted-foreground">
        <Loader2 className="mr-2 h-6 w-6 animate-spin" />
        Loading video details...
      </div>
    }>
      <VideoDetailContent params={params} />
    </Suspense>
  )
}

function DetectionDetail({ id, status }: { id: number; status: string }) {
  const statusColor = status === "Tracked" ? "text-primary" : status === "Requires Review" ? "text-destructive" : "text-accent"
  
  return (
    <div className="flex items-center justify-between p-2 rounded-lg bg-secondary/50 border border-border">
      <span className="text-sm text-foreground">Pedestrian ID #{id}</span>
      <span className={`text-xs ${statusColor}`}>{status}</span>
    </div>
  )
}
