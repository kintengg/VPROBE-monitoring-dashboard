"use client"

import { useMemo, useState } from "react"
import type { EventRecord, VideoSeverityBucket } from "@/lib/api"

interface PlaybackTimelineProps {
  startTime: string
  endTime: string
  durationSeconds: number
  currentTimeSeconds: number
  events: EventRecord[]
  severityBuckets?: VideoSeverityBucket[]
  searchMatchOffsets?: number[]
  onSeek?: (seconds: number) => void
}

type SeverityLevel = "neutral" | "light" | "moderate" | "heavy"

type ZoomPresetId = "full" | "30m" | "10m" | "5m"

const ZOOM_PRESETS: Array<{ id: ZoomPresetId; label: string; seconds: number | null }> = [
  { id: "full", label: "Full", seconds: null },
  { id: "30m", label: "30m", seconds: 30 * 60 },
  { id: "10m", label: "10m", seconds: 10 * 60 },
  { id: "5m", label: "5m", seconds: 5 * 60 },
]

const SEVERITY_STYLES: Record<SeverityLevel, { label: string; fill: string }> = {
  neutral: { label: "LOS A-B", fill: "rgba(16, 185, 129, 0.32)" },
  light: { label: "LOS C", fill: "rgba(132, 204, 22, 0.4)" },
  moderate: { label: "LOS D", fill: "rgba(245, 158, 11, 0.5)" },
  heavy: { label: "LOS E-F", fill: "rgba(239, 68, 68, 0.58)" },
}

function losFromScore(score?: number | null): string | null {
  if (typeof score !== "number" || !Number.isFinite(score)) return null
  if (score < 15) return "LOS A"
  if (score < 33) return "LOS B"
  if (score < 50) return "LOS C"
  if (score < 66) return "LOS D"
  if (score < 85) return "LOS E"
  return "LOS F"
}

function formatDuration(seconds: number) {
  const totalSeconds = Math.max(0, Math.floor(seconds))
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const remainingSeconds = totalSeconds % 60

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`
  }

  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`
}

function parseClockMinutes(value: string) {
  const trimmed = value.trim()
  const twelveHourMatch = trimmed.match(/^([0-9]{1,2}):([0-9]{2})(?::[0-9]{2})?\s*(AM|PM)$/i)

  if (twelveHourMatch) {
    let hours = Number(twelveHourMatch[1])
    const minutes = Number(twelveHourMatch[2])
    const period = twelveHourMatch[3].toUpperCase()

    if (period === "PM" && hours < 12) hours += 12
    if (period === "AM" && hours === 12) hours = 0
    return hours * 60 + minutes
  }

  const parts = trimmed.split(":")
  if (parts.length < 2) return null

  const hours = Number(parts[0])
  const minutes = Number(parts[1])
  if (!Number.isFinite(hours) || !Number.isFinite(minutes)) return null
  return hours * 60 + minutes
}

function formatClock(minutes: number) {
  const normalizedMinutes = ((Math.round(minutes) % 1440) + 1440) % 1440
  const hours24 = Math.floor(normalizedMinutes / 60)
  const mins = normalizedMinutes % 60
  const period = hours24 >= 12 ? "PM" : "AM"
  const hours12 = hours24 % 12 || 12
  return `${hours12}:${mins.toString().padStart(2, "0")} ${period}`
}

function clampOffset(offset: number, durationSeconds: number) {
  return Math.max(0, Math.min(offset, durationSeconds))
}

function formatRangeLabel(start: number, end: number) {
  if (Math.abs(end - start) < 1) {
    return formatDuration(start)
  }

  return `${formatDuration(start)}–${formatDuration(end)}`
}

function severityFromOcclusion(occlusionClass?: number | null): SeverityLevel {
  if (occlusionClass === 2) return "heavy"
  if (occlusionClass === 1) return "moderate"
  if (occlusionClass === 0) return "light"
  return "neutral"
}

function severityRank(level: SeverityLevel) {
  if (level === "heavy") return 3
  if (level === "moderate") return 2
  if (level === "light") return 1
  return 0
}

function adaptiveTickStep(windowDurationSeconds: number) {
  const target = Math.max(1, windowDurationSeconds / 6)
  const candidates = [30, 60, 120, 300, 600, 900, 1800, 3600]
  return candidates.find((candidate) => candidate >= target) ?? 3600
}

export function PlaybackTimeline({
  startTime,
  endTime,
  durationSeconds,
  currentTimeSeconds,
  events,
  severityBuckets: backendSeverityBuckets = [],
  searchMatchOffsets = [],
  onSeek,
}: PlaybackTimelineProps) {
  const [zoomPreset, setZoomPreset] = useState<ZoomPresetId>("full")
  const [hoveredOffsetSeconds, setHoveredOffsetSeconds] = useState<number | null>(null)
  const safeDuration = Number.isFinite(durationSeconds) && durationSeconds > 0 ? durationSeconds : 0
  const safeCurrentTime = Math.max(0, Math.min(currentTimeSeconds, safeDuration || currentTimeSeconds))
  const activeZoomConfig = ZOOM_PRESETS.find((preset) => preset.id === zoomPreset) ?? ZOOM_PRESETS[0]
  const zoomWindowDuration = safeDuration > 0
    ? Math.min(activeZoomConfig.seconds ?? safeDuration, safeDuration)
    : 0
  const zoomWindowStart = useMemo(() => {
    if (!safeDuration || !zoomWindowDuration || zoomWindowDuration >= safeDuration) {
      return 0
    }

    const maxStart = safeDuration - zoomWindowDuration
    const centered = safeCurrentTime - (zoomWindowDuration / 2)
    return Math.max(0, Math.min(centered, maxStart))
  }, [safeCurrentTime, safeDuration, zoomWindowDuration])
  const zoomWindowEnd = zoomWindowStart + zoomWindowDuration

  const projectRangeIntoWindow = (startOffset: number, endOffset: number) => {
    if (!zoomWindowDuration) return null

    const clippedStart = Math.max(startOffset, zoomWindowStart)
    const clippedEnd = Math.min(endOffset, zoomWindowEnd)
    if (clippedEnd <= clippedStart) return null

    return {
      left: ((clippedStart - zoomWindowStart) / zoomWindowDuration) * 100,
      width: ((clippedEnd - clippedStart) / zoomWindowDuration) * 100,
    }
  }

  const timedEvents = useMemo(
    () =>
      events
        .filter((event): event is EventRecord & { offsetSeconds: number } => typeof event.offsetSeconds === "number")
        .map((event) => ({
          ...event,
          offsetSeconds: clampOffset(event.offsetSeconds, safeDuration || event.offsetSeconds),
        }))
        .sort((left, right) => left.offsetSeconds - right.offsetSeconds),
    [events, safeDuration],
  )

  const summarizedSeverityBuckets = useMemo(() => {
    if (!safeDuration || !zoomWindowDuration || backendSeverityBuckets.length === 0) return []

    return backendSeverityBuckets
      .map((bucket) => {
        const startOffset = clampOffset(bucket.startOffsetSeconds, safeDuration)
        const endOffset = clampOffset(bucket.endOffsetSeconds, safeDuration)
        if (endOffset <= startOffset) return null
        const projected = projectRangeIntoWindow(startOffset, endOffset)
        if (!projected) return null

        const scoreLabel = typeof bucket.score === "number" ? ` • score ${bucket.score.toFixed(1)}` : ""
        return {
          startOffset,
          endOffset,
          left: projected.left,
          width: projected.width,
          severity: bucket.severity,
          score: typeof bucket.score === "number" ? bucket.score : null,
          title: `${SEVERITY_STYLES[bucket.severity].label}${scoreLabel} • ${formatRangeLabel(startOffset, endOffset)}`,
        }
      })
      .filter((bucket): bucket is { startOffset: number; endOffset: number; left: number; width: number; severity: SeverityLevel; score: number | null; title: string } => bucket !== null)
  }, [backendSeverityBuckets, safeDuration, zoomWindowDuration, zoomWindowEnd, zoomWindowStart])

  const fallbackSeverityBuckets = useMemo(() => {
    if (!safeDuration || !zoomWindowDuration) return []

    const classifiedEvents = timedEvents.filter(
      (event) => event.occlusionClass === 0 || event.occlusionClass === 1 || event.occlusionClass === 2,
    )

    if (classifiedEvents.length === 0) {
      return [
        {
          startOffset: 0,
          endOffset: safeDuration,
          left: 0,
          width: 100,
          severity: "neutral" as const,
          score: null as number | null,
          title: `No LOS samples • ${formatRangeLabel(0, safeDuration)}`,
        },
      ]
    }

    const rawSegments = classifiedEvents
      .map((event, index) => {
        const previous = classifiedEvents[index - 1]
        const next = classifiedEvents[index + 1]
        const startOffset = previous ? (previous.offsetSeconds + event.offsetSeconds) / 2 : 0
        const endOffset = next ? (event.offsetSeconds + next.offsetSeconds) / 2 : safeDuration

        return {
          startOffset: clampOffset(startOffset, safeDuration),
          endOffset: clampOffset(endOffset, safeDuration),
          severity: severityFromOcclusion(event.occlusionClass),
        }
      })
      .filter((segment) => segment.endOffset > segment.startOffset)

    const mergedSegments: Array<{ startOffset: number; endOffset: number; severity: SeverityLevel }> = []

    for (const segment of rawSegments) {
      const previous = mergedSegments[mergedSegments.length - 1]

      if (previous && previous.severity === segment.severity) {
        previous.endOffset = segment.endOffset
        continue
      }

      mergedSegments.push({ ...segment })
    }

    return mergedSegments
      .map((segment) => {
        const projected = projectRangeIntoWindow(segment.startOffset, segment.endOffset)
        if (!projected) return null

        return {
          startOffset: segment.startOffset,
          endOffset: segment.endOffset,
          left: projected.left,
          width: projected.width,
          severity: segment.severity,
          score: null as number | null,
          title: `${SEVERITY_STYLES[segment.severity].label} • ${formatRangeLabel(segment.startOffset, segment.endOffset)}`,
        }
      })
      .filter((segment): segment is { startOffset: number; endOffset: number; left: number; width: number; severity: SeverityLevel; score: number | null; title: string } => segment !== null)
  }, [safeDuration, timedEvents, zoomWindowDuration, zoomWindowEnd, zoomWindowStart])

  const severityBuckets = summarizedSeverityBuckets.length > 0 ? summarizedSeverityBuckets : fallbackSeverityBuckets

  const hoveredSeverity = useMemo(() => {
    if (hoveredOffsetSeconds === null || severityBuckets.length === 0) {
      return null
    }

    return severityBuckets.find(
      (bucket) => hoveredOffsetSeconds >= bucket.startOffset && hoveredOffsetSeconds <= bucket.endOffset,
    ) ?? null
  }, [hoveredOffsetSeconds, severityBuckets])

  const detectionBars = useMemo(() => {
    if (!zoomWindowDuration || timedEvents.length === 0) return []

    const windowedEvents = timedEvents.filter((event) => event.offsetSeconds >= zoomWindowStart && event.offsetSeconds <= zoomWindowEnd)
    if (windowedEvents.length === 0) return []

    const bySecond = new Map<number, { count: number; maxSeverity: SeverityLevel }>()
    for (const event of windowedEvents) {
      const severity = severityFromOcclusion(event.occlusionClass)
      const second = Math.floor(Math.max(0, event.offsetSeconds))
      const existing = bySecond.get(second)
      if (!existing) {
        bySecond.set(second, { count: 1, maxSeverity: severity })
        continue
      }

      existing.count += 1
      if (severityRank(severity) > severityRank(existing.maxSeverity)) {
        existing.maxSeverity = severity
      }
    }

    const bars = Array.from(bySecond.entries())
      .sort((left, right) => left[0] - right[0])
      .map(([second, summary]) => {
        const center = second + 0.5
        const clampedHeight = Math.min(22, 8 + (summary.count * 2))
        return {
          second,
          center,
          count: summary.count,
          maxSeverity: summary.maxSeverity,
          height: clampedHeight,
          left: ((center - zoomWindowStart) / zoomWindowDuration) * 100,
          title: `${formatDuration(second)}\n${summary.count} ${summary.count === 1 ? "event" : "events"}`,
        }
      })
      .filter((bar) => bar.left >= 0 && bar.left <= 100)

    return bars
  }, [timedEvents, zoomWindowDuration, zoomWindowEnd, zoomWindowStart])

  const searchClusters = useMemo(() => {
    if (!safeDuration || !zoomWindowDuration) return []

    const offsets = Array.from(
      new Set(
        searchMatchOffsets
          .filter((offset): offset is number => Number.isFinite(offset))
          .map((offset) => clampOffset(offset, safeDuration)),
      ),
    )
      .filter((offset) => offset >= zoomWindowStart && offset <= zoomWindowEnd)
      .sort((left, right) => left - right)

    if (offsets.length === 0) return []

    const mergeWindowSeconds = Math.max(4, safeDuration * 0.015)
    const rawClusters: Array<{ start: number; end: number; center: number; count: number }> = []

    for (const offset of offsets) {
      const previous = rawClusters[rawClusters.length - 1]
      if (previous && offset - previous.end <= mergeWindowSeconds) {
        previous.center = (previous.center * previous.count + offset) / (previous.count + 1)
        previous.end = offset
        previous.count += 1
        continue
      }

      rawClusters.push({ start: offset, end: offset, center: offset, count: 1 })
    }

    return rawClusters.map((cluster) => ({
      ...cluster,
      left: ((cluster.center - zoomWindowStart) / zoomWindowDuration) * 100,
      widthPercent: Math.max((((cluster.end - cluster.start) || mergeWindowSeconds * 0.6) / zoomWindowDuration) * 100, 0.9),
      title: `${formatRangeLabel(cluster.start, cluster.end)}\n${cluster.count} search ${cluster.count === 1 ? "match" : "matches"}`,
    }))
  }, [safeDuration, searchMatchOffsets, zoomWindowDuration, zoomWindowEnd, zoomWindowStart])

  const hasSearchMatches = searchClusters.length > 0

  const markerOffsets = useMemo(() => {
    if (!zoomWindowDuration) return []

    const step = adaptiveTickStep(zoomWindowDuration)
    const ticks: number[] = []
    const startTick = Math.ceil(zoomWindowStart / step) * step
    for (let tick = startTick; tick <= zoomWindowEnd; tick += step) {
      ticks.push(tick)
      if (ticks.length >= 8) break
    }

    if (ticks.length === 0 || ticks[0] !== zoomWindowStart) {
      ticks.unshift(zoomWindowStart)
    }
    if (ticks[ticks.length - 1] !== zoomWindowEnd) {
      ticks.push(zoomWindowEnd)
    }

    return ticks.map((offset, index) => ({
      id: `timeline-marker-${index}-${offset.toFixed(3)}`,
      label: (() => {
        const startMinutes = parseClockMinutes(startTime)
        if (startMinutes === null) {
          if (Math.abs(offset - zoomWindowEnd) < 0.1) {
            return endTime
          }
          return formatDuration(offset)
        }
        return formatClock(startMinutes + offset / 60)
      })(),
      leftPercent: ((offset - zoomWindowStart) / zoomWindowDuration) * 100,
    }))
  }, [endTime, startTime, zoomWindowDuration, zoomWindowEnd, zoomWindowStart])

  const currentPosition = zoomWindowDuration ? ((safeCurrentTime - zoomWindowStart) / zoomWindowDuration) * 100 : 0
  const currentWallClock = (() => {
    const startMinutes = parseClockMinutes(startTime)
    if (startMinutes === null) return formatDuration(safeCurrentTime)
    return formatClock(startMinutes + safeCurrentTime / 60)
  })()

  const handleSeek = (clientX: number, target: HTMLDivElement) => {
    if (!zoomWindowDuration || !onSeek) return
    const rect = target.getBoundingClientRect()
    const relativeX = Math.max(0, Math.min(clientX - rect.left, rect.width))
    onSeek(zoomWindowStart + ((relativeX / rect.width) * zoomWindowDuration))
  }

  const handleHover = (clientX: number, target: HTMLDivElement) => {
    if (!zoomWindowDuration) return
    const rect = target.getBoundingClientRect()
    const relativeX = Math.max(0, Math.min(clientX - rect.left, rect.width))
    setHoveredOffsetSeconds(zoomWindowStart + ((relativeX / rect.width) * zoomWindowDuration))
  }

  return (
    <div className="rounded-2xl border border-border bg-card p-5 shadow-elevated-sm">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium text-foreground">Playback Timeline</h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Click anywhere on the bar to jump through the recording.
            {hasSearchMatches ? " Search hits are highlighted in cyan." : ""}
          </p>
        </div>
        <div className="rounded-lg border border-border/70 bg-secondary/40 px-3 py-2 text-right text-xs text-muted-foreground">
          <p>{startTime} - {endTime}</p>
          <p className="mt-1 text-foreground">{currentWallClock}</p>
        </div>
      </div>

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">Zoom</span>
        {ZOOM_PRESETS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            className={`rounded-full border px-2.5 py-1 text-xs transition-colors ${
              zoomPreset === preset.id
                ? "border-primary bg-primary/15 text-primary"
                : "border-border bg-secondary/40 text-muted-foreground hover:bg-secondary"
            }`}
            onClick={() => setZoomPreset(preset.id)}
          >
            {preset.label}
          </button>
        ))}
        <span className="ml-auto rounded-full bg-secondary px-2.5 py-1 text-[11px] text-muted-foreground">
          Window {formatDuration(zoomWindowStart)} - {formatDuration(zoomWindowEnd)}
        </span>
        {hoveredOffsetSeconds !== null && (
          <span className="rounded-full border border-border bg-secondary/60 px-2.5 py-1 text-[11px] text-foreground">
            Hover {formatDuration(hoveredOffsetSeconds)} • {hoveredSeverity ? (losFromScore(hoveredSeverity.score) ?? SEVERITY_STYLES[hoveredSeverity.severity].label) : "No LOS"}
          </span>
        )}
      </div>

      {safeDuration > 0 && zoomWindowDuration > 0 ? (
        <>
          <div
            className="relative mb-2 h-16 cursor-pointer overflow-hidden rounded-xl border border-border bg-secondary/70"
            onClick={(event) => handleSeek(event.clientX, event.currentTarget)}
            onMouseMove={(event) => handleHover(event.clientX, event.currentTarget)}
            onMouseLeave={() => setHoveredOffsetSeconds(null)}
          >
            {severityBuckets.map((bucket) => (
              <div
                key={`${bucket.startOffset}-${bucket.endOffset}-${bucket.severity}`}
                className="absolute inset-y-0"
                style={{ left: `${bucket.left}%`, width: `${bucket.width}%`, backgroundColor: SEVERITY_STYLES[bucket.severity].fill }}
              />
            ))}

            <div className="pointer-events-none absolute inset-y-0 left-0 bg-background/10" style={{ width: `${currentPosition}%` }} />

            {searchClusters.map((cluster) => (
              <button
                key={`search-${cluster.start}-${cluster.count}`}
                type="button"
                aria-label={cluster.title}
                className="absolute bottom-1 z-10 h-2 rounded-full border border-cyan-200/80 bg-cyan-400/65 shadow-[0_0_0_1px_rgba(34,211,238,0.12)]"
                style={{ left: `${cluster.left}%`, width: `${cluster.widthPercent}%`, transform: "translateX(-50%)" }}
                onClick={(event) => {
                  event.stopPropagation()
                  onSeek?.(cluster.center)
                }}
              />
            ))}

            {detectionBars.map((bar) => {
              const markerStyle = bar.maxSeverity === "heavy"
                ? "bg-red-500/95"
                : bar.maxSeverity === "moderate"
                  ? "bg-amber-500/95"
                  : bar.maxSeverity === "light"
                    ? "bg-lime-500/95"
                    : "bg-emerald-500/95"

              return (
                <button
                  key={`marker-${bar.second}`}
                  type="button"
                  aria-label={bar.title}
                  className={`absolute bottom-1 z-20 w-1.5 -translate-x-1/2 rounded-[2px] shadow-sm transition-transform hover:scale-y-110 ${markerStyle}`}
                  style={{
                    left: `${bar.left}%`,
                    height: `${bar.height}px`,
                  }}
                  onClick={(event) => {
                    event.stopPropagation()
                    onSeek?.(bar.second)
                  }}
                />
              )
            })}

            <div className="absolute inset-y-0 z-10 w-0.5 bg-accent" style={{ left: `${currentPosition}%` }}>
              <div className="absolute -top-1 left-1/2 h-3 w-3 -translate-x-1/2 rounded-full border border-background bg-accent" />
            </div>
          </div>

          <input
            type="range"
            min={zoomWindowStart}
            max={zoomWindowEnd}
            step={0.1}
            value={safeCurrentTime}
            className="mt-1 h-2.5 w-full cursor-pointer accent-primary"
            onChange={(event) => onSeek?.(Number(event.target.value))}
          />

          <div className="relative mt-3 h-4 text-[10px] text-muted-foreground">
            {markerOffsets.map((marker) => (
              <span
                key={marker.id}
                className="absolute -translate-x-1/2 truncate"
                style={{ left: `${marker.leftPercent}%`, maxWidth: "90px" }}
              >
                {marker.label}
              </span>
            ))}
          </div>

          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-3 text-xs text-muted-foreground">
            <div className="flex flex-wrap items-center gap-3">
              <span className="inline-flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-sm bg-emerald-500" />
                LOS A-B
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-sm bg-lime-500" />
                LOS C
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-sm bg-amber-500" />
                LOS D
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-sm bg-red-500" />
                LOS E-F
              </span>
            </div>
            <span>
              {formatDuration(safeCurrentTime)} / {formatDuration(safeDuration)}
            </span>
          </div>
        </>
      ) : (
        <div className="rounded-xl border border-dashed border-border bg-secondary/40 px-4 py-6 text-sm text-muted-foreground">
          Timeline controls will activate once the video metadata loads.
        </div>
      )}
    </div>
  )
}
