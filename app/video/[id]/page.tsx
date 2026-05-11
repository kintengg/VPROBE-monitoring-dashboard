"use client"

import { Suspense, use, useEffect, useMemo, useRef, useState } from "react"
import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation"
import { VideoPlayer } from "@/components/video/video-player"
import { PlaybackTimeline } from "@/components/video/playback-timeline"
import { EventFeed } from "@/components/surveillance/event-feed"
import { AISearchBar } from "@/components/surveillance/ai-search-bar"
import { Button } from "@/components/ui/button"
import { deleteVideo, getEvents, getLocations, getMediaUrl, getVideo, getVideoPlaybackPath, type EventRecord, type LocationRecord, type VideoDetailRecord, type VideoPedestrianTrackRecord, type VideoSeverityBucket } from "@/lib/api"
import { useSetVideoDomain } from "@/components/video-domain-context"

type LOSLevel = "A" | "B" | "C" | "D" | "E" | "F"

function losFromScore(score?: number | null): LOSLevel | null {
  if (typeof score !== "number" || !Number.isFinite(score)) return null
  if (score < 15) return "A"
  if (score < 33) return "B"
  if (score < 50) return "C"
  if (score < 66) return "D"
  if (score < 85) return "E"
  return "F"
}

function losScore(level: LOSLevel | null): number {
  if (level === "A") return 0
  if (level === "B") return 1
  if (level === "C") return 2
  if (level === "D") return 3
  if (level === "E") return 4
  if (level === "F") return 5
  return -1
}

function formatLos(level: LOSLevel | null): string {
  return level ? `LOS ${level}` : "LOS --"
}

function losFromRollingDetections(countInLastMinute: number): LOSLevel {
  if (countInLastMinute <= 0) return "A"
  if (countInLastMinute <= 2) return "B"
  if (countInLastMinute <= 4) return "C"
  if (countInLastMinute <= 7) return "D"
  if (countInLastMinute <= 10) return "E"
  return "F"
}

interface PortableTimelineRow {
  offsetSeconds: number
  severity?: "neutral" | "light" | "moderate" | "heavy" | null
  ptsiScore?: number | null
  los?: LOSLevel | null
  detectedNow?: number | null
  visiblePedestrians?: number | null
  totalPedestriansSoFar?: number | null
  cumulativeUniquePedestrians?: number | null
}

function parseClockToSeconds(value: string): number | null {
  const parts = value.trim().split(":").map((part) => Number(part))
  if (parts.length < 2 || parts.some((part) => !Number.isFinite(part))) {
    return null
  }

  if (parts.length === 3) {
    return (parts[0] * 3600) + (parts[1] * 60) + parts[2]
  }

  return (parts[0] * 60) + parts[1]
}

function inferOffsetFromTimestamp(timestamp?: string | null, startTime?: string | null): number | null {
  if (!timestamp) return null

  const eventSeconds = parseClockToSeconds(timestamp)
  if (eventSeconds === null) return null

  const startSeconds = parseClockToSeconds(startTime ?? "")
  if (startSeconds === null) {
    return Math.max(0, eventSeconds)
  }

  const normalizedEnd = eventSeconds >= startSeconds ? eventSeconds : eventSeconds + (24 * 3600)
  return Math.max(0, normalizedEnd - startSeconds)
}

function normalizeLos(level: unknown): LOSLevel | null {
  if (level === "A" || level === "B" || level === "C" || level === "D" || level === "E" || level === "F") {
    return level
  }
  return null
}

function timelineSeverityFromScore(score?: number | null): "neutral" | "light" | "moderate" | "heavy" {
  if (typeof score !== "number" || !Number.isFinite(score)) return "neutral"
  if (score < 33) return "light"
  if (score < 66) return "moderate"
  return "heavy"
}

function durationFromClockRange(startTime: string, endTime: string): number {
  const startSeconds = parseClockToSeconds(startTime)
  const endSeconds = parseClockToSeconds(endTime)
  if (startSeconds === null || endSeconds === null) return 0
  const normalizedEnd = endSeconds >= startSeconds ? endSeconds : endSeconds + (24 * 3600)
  return Math.max(0, normalizedEnd - startSeconds)
}

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
  const [currentTimeSeconds, setCurrentTimeSeconds] = useState(0)
  const [durationSeconds, setDurationSeconds] = useState(0)
  const [portableTimelineRows, setPortableTimelineRows] = useState<PortableTimelineRow[]>([])
  const [deleting, setDeleting] = useState(false)
  const videoElementRef = useRef<HTMLVideoElement | null>(null)
  const seekTokenRef = useRef(0)
  const appliedQuerySeekRef = useRef("")

  const setVideoDomain = useSetVideoDomain()
  useEffect(() => {
    setVideoDomain(videoLocation?.domain ?? null)
  }, [videoLocation?.domain, setVideoDomain])

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
    setCurrentTimeSeconds(0)
    setDurationSeconds(0)
    setPortableTimelineRows([])
  }, [id])

  useEffect(() => {
    let cancelled = false

    const loadPortableTimeline = async () => {
      try {
        const timelineUrl = getMediaUrl(`storage/portable/videos/${id}/timeline.json`)
        if (!timelineUrl) {
          if (!cancelled) setPortableTimelineRows([])
          return
        }

        const maxAttempts = 4
        const baseBackoffMs = 250

        for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
          if (cancelled) return

          try {
            const response = await fetch(timelineUrl, { cache: "no-store" })
            if (!response.ok) {
              throw new Error(`Timeline fetch failed with status ${response.status}`)
            }

            const payload = await response.json()
            if (!Array.isArray(payload)) {
              if (!cancelled) setPortableTimelineRows([])
              return
            }

            const rows = payload
              .map((row): PortableTimelineRow | null => {
                const offsetRaw = Number((row as Record<string, unknown>).offsetSeconds)
                if (!Number.isFinite(offsetRaw)) return null

                const scoreValue = Number((row as Record<string, unknown>).ptsiScore)
                const detectedNowValue = Number((row as Record<string, unknown>).detectedNow)
                const visibleNowValue = Number((row as Record<string, unknown>).visiblePedestrians)
                const totalSoFarValue = Number((row as Record<string, unknown>).totalPedestriansSoFar)
                const cumulativeValue = Number((row as Record<string, unknown>).cumulativeUniquePedestrians)
                const score = Number.isFinite(scoreValue) ? scoreValue : null

                return {
                  offsetSeconds: Math.max(0, offsetRaw),
                  severity: ((row as Record<string, unknown>).severity as PortableTimelineRow["severity"]) ?? null,
                  ptsiScore: score,
                  los: normalizeLos((row as Record<string, unknown>).los),
                  detectedNow: Number.isFinite(detectedNowValue) ? detectedNowValue : null,
                  visiblePedestrians: Number.isFinite(visibleNowValue) ? visibleNowValue : null,
                  totalPedestriansSoFar: Number.isFinite(totalSoFarValue) ? totalSoFarValue : null,
                  cumulativeUniquePedestrians: Number.isFinite(cumulativeValue) ? cumulativeValue : null,
                }
              })
              .filter((row): row is PortableTimelineRow => row !== null)
              .sort((left, right) => left.offsetSeconds - right.offsetSeconds)

            if (!cancelled) {
              setPortableTimelineRows(rows)
            }
            return
          } catch {
            if (attempt >= maxAttempts) {
              if (!cancelled) {
                setPortableTimelineRows([])
              }
              return
            }

            const backoffMs = baseBackoffMs * attempt
            await new Promise((resolve) => {
              setTimeout(resolve, backoffMs)
            })
          }
        }
      } catch {
        if (!cancelled) {
          setPortableTimelineRows([])
        }
      }
    }

    void loadPortableTimeline()

    return () => {
      cancelled = true
    }
  }, [id, video?.processedPath])

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
    seekTokenRef.current += 1
    setRequestedSeek({ seconds, token: seekTokenRef.current })
  }, [id, searchParams])

  const normalizedEvents = useMemo(
    () =>
      events.map((event) => {
        if (typeof event.offsetSeconds === "number" && Number.isFinite(event.offsetSeconds)) {
          return event
        }

        const inferredOffset = inferOffsetFromTimestamp(event.timestamp, video?.startTime)
        return inferredOffset === null ? event : { ...event, offsetSeconds: inferredOffset }
      }),
    [events, video?.startTime],
  )

  const orderedEvents = useMemo(
    () =>
      [...normalizedEvents].sort((left, right) => {
        const leftOffset = typeof left.offsetSeconds === "number" ? left.offsetSeconds : Number.POSITIVE_INFINITY
        const rightOffset = typeof right.offsetSeconds === "number" ? right.offsetSeconds : Number.POSITIVE_INFINITY
        return leftOffset - rightOffset
      }),
    [normalizedEvents],
  )

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

  const vehicleClassSummary = useMemo(() => {
    const classPatterns: Array<{ label: string; pattern: RegExp }> = [
      { label: "Jeepney", pattern: /\bjeepney\b/i },
      { label: "Tricycle", pattern: /\btricycle\b/i },
      { label: "Motorcycle", pattern: /\bmotorcycle\b|\bmotorbike\b/i },
      { label: "Bicycle", pattern: /\bbicycle\b|\bbike\b/i },
      { label: "Truck", pattern: /\btruck\b/i },
      { label: "Bus", pattern: /\bbus\b/i },
      { label: "Van", pattern: /\bvan\b/i },
      { label: "SUV", pattern: /\bsuv\b/i },
      { label: "Car", pattern: /\bcar\b|\bauto\b|\bautomobile\b/i },
    ]

    const formatVehicleClassLabel = (value: string) =>
      value
        .split(/[-_\s]+/)
        .filter(Boolean)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
        .join(" ")

    const detectionEvents = orderedEvents.filter((event) => event.type === "detection")
    const counts = new Map<string, number>()

    for (const event of detectionEvents) {
      const backendLabel = (event.vehicleClassLabel ?? "").trim()
      const backendClass = (event.vehicleClass ?? "").trim()
      let classLabel = backendLabel || (backendClass ? formatVehicleClassLabel(backendClass) : "")

      if (!classLabel) {
        const description = event.description ?? ""
        const matched = classPatterns.find((item) => item.pattern.test(description))
        classLabel = matched?.label ?? "Unclassified"
      }

      counts.set(classLabel, (counts.get(classLabel) ?? 0) + 1)
    }

    const classRows = Array.from(counts.entries())
      .map(([label, count]) => ({ label, count }))
      .sort((left, right) => right.count - left.count)

    const uniqueVehicleTracks = new Set(
      detectionEvents
        .map((event) => event.pedestrianId)
        .filter((pedestrianId): pedestrianId is number => typeof pedestrianId === "number"),
    ).size

    return {
      rows: classRows,
      totalDetections: detectionEvents.length,
      uniqueVehicleTracks,
    }
  }, [orderedEvents])

  const vehiclePlaybackWindows = useMemo(() => {
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

  const timelineRowsBySecond = useMemo(() => {
    const bySecond = new Map<number, PortableTimelineRow>()
    for (const row of portableTimelineRows) {
      bySecond.set(Math.floor(Math.max(0, row.offsetSeconds)), row)
    }
    return bySecond
  }, [portableTimelineRows])

  const hasTimelineCountSignal = useMemo(
    () =>
      portableTimelineRows.some((row) => {
        const visibleNow = typeof row.visiblePedestrians === "number" ? row.visiblePedestrians : 0
        const detectedNow = typeof row.detectedNow === "number" ? row.detectedNow : 0
        const totalSoFar = typeof row.totalPedestriansSoFar === "number" ? row.totalPedestriansSoFar : 0
        const cumulative = typeof row.cumulativeUniquePedestrians === "number" ? row.cumulativeUniquePedestrians : 0
        return visibleNow > 0 || detectedNow > 0 || totalSoFar > 0 || cumulative > 0
      }),
    [portableTimelineRows],
  )

  const hasTimelineLosSignal = useMemo(
    () =>
      portableTimelineRows.some(
        (row) => row.los !== null || (typeof row.ptsiScore === "number" && Number.isFinite(row.ptsiScore) && row.ptsiScore > 0),
      ),
    [portableTimelineRows],
  )

  const hasBackendLosSignal = useMemo(() => {
    const buckets = video?.severitySummary?.buckets ?? []
    return buckets.some(
      (bucket) => bucket.severity !== "neutral" || (typeof bucket.score === "number" && Number.isFinite(bucket.score) && bucket.score > 0),
    )
  }, [video?.severitySummary?.buckets])

  const proxyLosSummary = useMemo(() => {
    const starts = vehiclePlaybackWindows
      .map((window) => window.start)
      .filter((value) => Number.isFinite(value) && value >= 0)
      .sort((left, right) => left - right)

    if (starts.length === 0) {
      return { current: null as LOSLevel | null, worst: null as LOSLevel | null, average: null as LOSLevel | null }
    }

    const windowSeconds = 60
    const countRecent = (time: number) => {
      const lowerBound = time - windowSeconds
      let count = 0
      for (const start of starts) {
        if (start > time) break
        if (start >= lowerBound) count += 1
      }
      return count
    }

    const current = losFromRollingDetections(countRecent(Math.max(0, currentTimeSeconds)))

    const sampleUpperBound = Math.max(
      Math.ceil(durationSeconds > 0 ? durationSeconds : starts[starts.length - 1] + 1),
      Math.ceil(starts[starts.length - 1] + 1),
    )

    const sampledLos: LOSLevel[] = []
    for (let second = 0; second <= sampleUpperBound; second += 1) {
      sampledLos.push(losFromRollingDetections(countRecent(second)))
    }

    const worst = sampledLos.reduce<LOSLevel | null>((acc, level) => (losScore(level) > losScore(acc) ? level : acc), null)
    const averageRank = sampledLos.reduce((sum, level) => sum + losScore(level), 0) / sampledLos.length
    const average: LOSLevel = averageRank < 0.5
      ? "A"
      : averageRank < 1.5
        ? "B"
        : averageRank < 2.5
          ? "C"
          : averageRank < 3.5
            ? "D"
            : averageRank < 4.5
              ? "E"
              : "F"

    return { current, worst, average }
  }, [currentTimeSeconds, durationSeconds, vehiclePlaybackWindows])

  const activeTimelineRow = useMemo(
    () => timelineRowsBySecond.get(Math.floor(Math.max(0, currentTimeSeconds))) ?? null,
    [currentTimeSeconds, timelineRowsBySecond],
  )

  const trackedVehiclesSoFar = useMemo(() => {
    const timelineValue = hasTimelineCountSignal
      ? activeTimelineRow?.totalPedestriansSoFar ?? activeTimelineRow?.cumulativeUniquePedestrians
      : null
    if (typeof timelineValue === "number" && Number.isFinite(timelineValue) && timelineValue >= 0) {
      return Math.max(0, Math.round(timelineValue))
    }

    return vehiclePlaybackWindows.reduce((count, window) => (window.start <= currentTimeSeconds ? count + 1 : count), 0)
  }, [activeTimelineRow, currentTimeSeconds, hasTimelineCountSignal, vehiclePlaybackWindows])

  const liveDetectedVehicles = useMemo(
    () => {
      const timelineValue = hasTimelineCountSignal
        ? activeTimelineRow?.detectedNow ?? activeTimelineRow?.visiblePedestrians
        : null
      if (typeof timelineValue === "number" && Number.isFinite(timelineValue) && timelineValue >= 0) {
        return Math.max(0, Math.round(timelineValue))
      }

      return vehiclePlaybackWindows.reduce(
        (count, window) => (currentTimeSeconds >= window.start && currentTimeSeconds <= window.end ? count + 1 : count),
        0,
      )
    },
    [activeTimelineRow, currentTimeSeconds, hasTimelineCountSignal, vehiclePlaybackWindows],
  )

  const losSummary = useMemo(() => {
    if (portableTimelineRows.length > 0 && hasTimelineLosSignal) {
      const current = activeTimelineRow?.los ?? losFromScore(activeTimelineRow?.ptsiScore)
      const scoredLevels = portableTimelineRows
        .map((row) => row.los ?? losFromScore(row.ptsiScore))
        .filter((level): level is LOSLevel => level !== null)

      const worst = scoredLevels.reduce<LOSLevel | null>((acc, level) => (losScore(level) > losScore(acc) ? level : acc), null)

      const scoredValues = portableTimelineRows
        .map((row) => (typeof row.ptsiScore === "number" ? row.ptsiScore : null))
        .filter((score): score is number => score !== null)

      const average = scoredValues.length > 0
        ? losFromScore(scoredValues.reduce((sum, value) => sum + value, 0) / scoredValues.length)
        : null

      return { current, worst, average }
    }

    if (!hasBackendLosSignal) {
      return proxyLosSummary
    }

    const buckets = video?.severitySummary?.buckets ?? []
    if (buckets.length === 0) {
      return { current: null as LOSLevel | null, worst: null as LOSLevel | null, average: null as LOSLevel | null }
    }

    const bucketAtCurrent = buckets.find(
      (bucket) => currentTimeSeconds >= bucket.startOffsetSeconds && currentTimeSeconds <= bucket.endOffsetSeconds,
    )
    const current = losFromScore(bucketAtCurrent?.score)

    const scoredLevels = buckets
      .map((bucket) => losFromScore(bucket.score))
      .filter((level): level is LOSLevel => level !== null)

    const worst = scoredLevels.reduce<LOSLevel | null>((acc, level) => (losScore(level) > losScore(acc) ? level : acc), null)

    const scoredValues = buckets
      .map((bucket) => (typeof bucket.score === "number" ? bucket.score : null))
      .filter((score): score is number => score !== null)

    const average = scoredValues.length > 0
      ? losFromScore(scoredValues.reduce((sum, value) => sum + value, 0) / scoredValues.length)
      : null

    return { current, worst, average }
  }, [
    activeTimelineRow,
    currentTimeSeconds,
    hasBackendLosSignal,
    hasTimelineLosSignal,
    portableTimelineRows,
    proxyLosSummary,
    video?.severitySummary?.buckets,
  ])

  const playbackSeverityBuckets = useMemo<VideoSeverityBucket[]>(() => {
    const backendBuckets = video?.severitySummary?.buckets ?? []
    if (backendBuckets.length > 0) {
      return backendBuckets
    }

    if (portableTimelineRows.length < 2) {
      return []
    }

    const buckets: VideoSeverityBucket[] = []
    let segmentStart = portableTimelineRows[0].offsetSeconds
    let currentSeverity = portableTimelineRows[0].severity ?? timelineSeverityFromScore(portableTimelineRows[0].ptsiScore)
    let runningScoreTotal = typeof portableTimelineRows[0].ptsiScore === "number" ? portableTimelineRows[0].ptsiScore : 0
    let runningScoreCount = typeof portableTimelineRows[0].ptsiScore === "number" ? 1 : 0

    for (let index = 1; index < portableTimelineRows.length; index += 1) {
      const row = portableTimelineRows[index]
      const rowSeverity = row.severity ?? timelineSeverityFromScore(row.ptsiScore)
      const changedSeverity = rowSeverity !== currentSeverity

      if (changedSeverity) {
        buckets.push({
          startOffsetSeconds: segmentStart,
          endOffsetSeconds: row.offsetSeconds,
          severity: currentSeverity,
          score: runningScoreCount > 0 ? runningScoreTotal / runningScoreCount : null,
        })

        segmentStart = row.offsetSeconds
        currentSeverity = rowSeverity
        runningScoreTotal = 0
        runningScoreCount = 0
      }

      if (typeof row.ptsiScore === "number") {
        runningScoreTotal += row.ptsiScore
        runningScoreCount += 1
      }
    }

    const finalOffset = portableTimelineRows[portableTimelineRows.length - 1].offsetSeconds + 1
    buckets.push({
      startOffsetSeconds: segmentStart,
      endOffsetSeconds: finalOffset,
      severity: currentSeverity,
      score: runningScoreCount > 0 ? runningScoreTotal / runningScoreCount : null,
    })

    return buckets
  }, [portableTimelineRows, video?.severitySummary?.buckets])

  const fallbackDurationSeconds = useMemo(() => {
    const timelineMaxOffset = portableTimelineRows.length > 0
      ? portableTimelineRows[portableTimelineRows.length - 1].offsetSeconds + 1
      : 0
    const clockRangeDuration = video ? durationFromClockRange(video.startTime, video.endTime) : 0
    return Math.max(timelineMaxOffset, clockRangeDuration)
  }, [portableTimelineRows, video])

  const effectiveDurationSeconds = durationSeconds > 0 ? durationSeconds : fallbackDurationSeconds

  const mediaUrl = video ? getMediaUrl(getVideoPlaybackPath(video)) : null
  const metricTiles = [
    {
      icon: Car,
      value: String(trackedVehiclesSoFar),
      caption: "Total so far",
      iconClassName: "bg-emerald-500 text-white ring-emerald-500/20",
      valueClassName: "text-emerald-100 sm:text-foreground",
      cardClassName: "border-emerald-400/30 bg-gradient-to-br from-emerald-500/18 via-green-500/10 to-transparent",
    },
    {
      icon: Gauge,
      value: formatLos(losSummary.current),
      caption: "Current LOS",
      iconClassName: "bg-violet-500/90 text-white ring-violet-500/20",
      valueClassName: "text-foreground",
      cardClassName: "border-violet-400/25 bg-gradient-to-br from-violet-500/12 via-fuchsia-500/8 to-transparent",
    },
    {
      icon: OctagonAlert,
      value: formatLos(losSummary.worst),
      caption: "Worst LOS",
      iconClassName: "bg-amber-500/90 text-white ring-amber-500/20",
      valueClassName: "text-foreground",
      cardClassName: "border-amber-400/25 bg-gradient-to-br from-amber-500/12 via-orange-500/8 to-transparent",
    },
    {
      icon: BarChart3,
      value: formatLos(losSummary.average),
      caption: "Average LOS",
      iconClassName: "bg-sky-500/90 text-white ring-sky-500/20",
      valueClassName: "text-foreground",
      cardClassName: "border-sky-400/25 bg-gradient-to-br from-sky-500/12 via-cyan-500/8 to-transparent",
    },
  ]

  const requestSeek = (seconds: number) => {
    setCurrentTimeSeconds(Math.max(0, seconds))
    seekTokenRef.current += 1
    setRequestedSeek({ seconds, token: seekTokenRef.current })
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
              isVehicle={isVehicle}
              timestamp={video.timestamp}
              date={video.date}
              isProcessed={Boolean(video.processedPath)}
              videoRef={videoElementRef}
              requestedSeek={requestedSeek}
              roiCoordinates={videoLocation?.roiCoordinates ?? null}
              showROI={false}
              onTimeUpdate={setCurrentTimeSeconds}
              onDurationChange={setDurationSeconds}
            />

            <div className="flex flex-nowrap gap-3 overflow-x-auto pb-1">
              {metricTiles.map((tile) => {
                const Icon = tile.icon
                return (
                  <div
                    key={tile.caption}
                    className={`flex min-w-[220px] flex-1 items-center gap-3 rounded-2xl border px-4 py-3 shadow-elevated-sm ${tile.cardClassName}`}
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
              <div className="rounded-xl border border-border bg-card p-4">
                <PlaybackTimeline
                  startTime={video.startTime}
                  endTime={video.endTime}
                  durationSeconds={effectiveDurationSeconds}
                  currentTimeSeconds={currentTimeSeconds}
                  events={orderedEvents}
                  searchMatchOffsets={searchMatchOffsets}
                  onSeek={requestSeek}
                />
              </div>
            
          </div>
        </div>
      </div>

      {/* Right Sidebar - Filtered Event Feed */}
      <aside className="w-80 border-l border-border bg-card flex flex-col h-full">
        <AISearchBar />

        {/* Vehicle Class Summary */}
        <div className="border-b border-border p-4">
          <h4 className="mb-3 text-sm font-medium text-foreground">Vehicle Class Summary</h4>
          {vehicleClassSummary.totalDetections > 0 ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="rounded-lg border border-border bg-secondary/40 px-2.5 py-2">
                  <p className="text-muted-foreground">Detections</p>
                  <p className="mt-1 text-sm font-semibold text-foreground">{vehicleClassSummary.totalDetections}</p>
                </div>
                <div className="rounded-lg border border-border bg-secondary/40 px-2.5 py-2">
                  <p className="text-muted-foreground">Unique Tracks</p>
                  <p className="mt-1 text-sm font-semibold text-foreground">{vehicleClassSummary.uniqueVehicleTracks}</p>
                </div>
              </div>

              <div className="space-y-2">
                {vehicleClassSummary.rows.map((item) => {
                  const percentage = vehicleClassSummary.totalDetections > 0
                    ? Math.round((item.count / vehicleClassSummary.totalDetections) * 100)
                    : 0

                  return (
                    <div key={item.label} className="rounded-lg border border-border bg-secondary/30 p-2">
                      <div className="mb-1 flex items-center justify-between text-xs">
                        <span className="font-medium text-foreground">{item.label}</span>
                        <span className="text-muted-foreground">{item.count} ({percentage}%)</span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
                        <div className="h-full rounded-full bg-primary" style={{ width: `${percentage}%` }} />
                      </div>
                    </div>
                  )
                })}
              </div>

              {vehicleClassSummary.rows.length === 1 && vehicleClassSummary.rows[0]?.label === "Unclassified" ? (
                <p className="text-[11px] text-muted-foreground">
                  Class labels are not present in this video&apos;s event payload, so detections are grouped as unclassified.
                </p>
              ) : null}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No detection events are available yet for class summarization.</p>
          )}
        </div>

        <EventFeed
          filteredVideoId={id}
          events={orderedEvents}
          loading={loading}
          selectedEventId={selectedEventId}
          onEventSelect={handleEventSelect}
        />
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
