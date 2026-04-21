"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { AlertCircle, MapPin, Users } from "lucide-react"
import type { GateDirectionConfiguration, ROIConfiguration } from "@/lib/api"

interface VideoPlayerProps {
  videoId: string
  location: string
  src?: string | null
  pedestrianCount: number
  timestamp: string
  date: string
  isProcessed: boolean
  videoRef?: { current: HTMLVideoElement | null }
  requestedSeek?: { seconds: number; token: number } | null
  roiCoordinates?: ROIConfiguration | null
  showROI?: boolean
  entryExitPoints?: GateDirectionConfiguration | null
  showEntryExitPoints?: boolean
  onTimeUpdate?: (seconds: number) => void
  onDurationChange?: (seconds: number) => void
}

const DIRECTIONAL_STRIP_STYLES = [
  { key: "strip_0", label: "strip_0", stroke: "#22d3ee" },
  { key: "strip_1", label: "strip_1", stroke: "#fde047" },
  { key: "strip_2", label: "strip_2", stroke: "#ff4dff" },
] as const

function applySeek(video: HTMLVideoElement, seconds: number) {
  const nextTime = Number.isFinite(video.duration) ? Math.min(Math.max(seconds, 0), video.duration) : Math.max(seconds, 0)
  video.currentTime = nextTime
  return nextTime
}

function toOverlayPointString(polygon: Array<[number, number]>, width: number, height: number) {
  return polygon.map(([x, y]) => `${x * width},${y * height}`).join(" ")
}

function polygonCentroid(polygon: Array<[number, number]>, width: number, height: number) {
  if (polygon.length === 0) {
    return { x: 0, y: 0 }
  }

  const total = polygon.reduce(
    (accumulator, [x, y]) => ({ x: accumulator.x + (x * width), y: accumulator.y + (y * height) }),
    { x: 0, y: 0 },
  )

  return {
    x: total.x / polygon.length,
    y: total.y / polygon.length,
  }
}

export function VideoPlayer({
  videoId,
  location,
  src,
  pedestrianCount,
  timestamp,
  date,
  isProcessed,
  videoRef,
  requestedSeek,
  roiCoordinates,
  showROI = false,
  entryExitPoints,
  showEntryExitPoints = false,
  onTimeUpdate,
  onDurationChange,
}: VideoPlayerProps) {
  const fallbackRef = useRef<HTMLVideoElement | null>(null)
  const frameRef = useRef<HTMLDivElement | null>(null)
  const resolvedRef = videoRef ?? fallbackRef
  const [overlaySize, setOverlaySize] = useState({ width: 0, height: 0 })
  const roiPolygons = roiCoordinates?.includePolygonsNorm ?? []
  const directionalZones = entryExitPoints?.gateDirectionZonesNorm ?? null

  useEffect(() => {
    const element = frameRef.current
    if (!element || typeof ResizeObserver === "undefined") {
      return
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) {
        return
      }

      setOverlaySize({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      })
    })

    observer.observe(element)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!src) {
      onDurationChange?.(0)
      onTimeUpdate?.(0)
    }
  }, [onDurationChange, onTimeUpdate, src])

  useEffect(() => {
    if (!requestedSeek || !resolvedRef.current) {
      return
    }

    const video = resolvedRef.current
    const seekToRequestedTime = () => onTimeUpdate?.(applySeek(video, requestedSeek.seconds))

    if (video.readyState >= 1) {
      seekToRequestedTime()
      return
    }

    video.addEventListener("loadedmetadata", seekToRequestedTime, { once: true })
    return () => video.removeEventListener("loadedmetadata", seekToRequestedTime)
  }, [requestedSeek, resolvedRef, src])

  useEffect(() => {
    const video = resolvedRef.current
    if (!video || !src) {
      return
    }

    let frameId: number | null = null

    const publishCurrentTime = () => onTimeUpdate?.(video.currentTime)
    const publishDuration = () => onDurationChange?.(Number.isFinite(video.duration) ? video.duration : 0)
    const stopFrameLoop = () => {
      if (frameId !== null) {
        cancelAnimationFrame(frameId)
        frameId = null
      }
    }
    const frameLoop = () => {
      publishCurrentTime()
      if (!video.paused && !video.ended) {
        frameId = requestAnimationFrame(frameLoop)
      } else {
        frameId = null
      }
    }
    const startFrameLoop = () => {
      stopFrameLoop()
      frameLoop()
    }
    const handleLoadedMetadata = () => {
      publishDuration()
      publishCurrentTime()
    }

    video.addEventListener("loadedmetadata", handleLoadedMetadata)
    video.addEventListener("play", startFrameLoop)
    video.addEventListener("pause", publishCurrentTime)
    video.addEventListener("ended", publishCurrentTime)
    video.addEventListener("seeking", publishCurrentTime)
    video.addEventListener("seeked", publishCurrentTime)

    publishDuration()
    publishCurrentTime()
    if (!video.paused && !video.ended) {
      startFrameLoop()
    }

    return () => {
      stopFrameLoop()
      video.removeEventListener("loadedmetadata", handleLoadedMetadata)
      video.removeEventListener("play", startFrameLoop)
      video.removeEventListener("pause", publishCurrentTime)
      video.removeEventListener("ended", publishCurrentTime)
      video.removeEventListener("seeking", publishCurrentTime)
      video.removeEventListener("seeked", publishCurrentTime)
    }
  }, [onDurationChange, onTimeUpdate, resolvedRef, src])

  const overlayWidth = overlaySize.width > 0 ? overlaySize.width : 1
  const overlayHeight = overlaySize.height > 0 ? overlaySize.height : 1
  const showAnyOverlay = (showROI && roiPolygons.length > 0) || (showEntryExitPoints && directionalZones != null)
  const directionalZoneOverlays = useMemo(() => {
    if (!directionalZones) {
      return []
    }

    return DIRECTIONAL_STRIP_STYLES.map(({ key, label, stroke }) => {
      const polygon = directionalZones[key]
      return {
        key,
        label,
        stroke,
        points: toOverlayPointString(polygon, overlayWidth, overlayHeight),
        centroid: polygonCentroid(polygon, overlayWidth, overlayHeight),
      }
    })
  }, [directionalZones, overlayHeight, overlayWidth])
  const overlayStrokeWidth = Math.max(2, Math.min(overlayWidth, overlayHeight) * 0.004)
  const labelFontSize = Math.max(10, Math.min(13, Math.min(overlayWidth, overlayHeight) * 0.018))

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-card shadow-elevated-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3">
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className={`rounded-full px-2.5 py-1 font-medium ${isProcessed ? "bg-primary/15 text-primary" : src ? "bg-accent/15 text-accent" : "bg-muted text-muted-foreground"}`}>
            {isProcessed ? "Annotated output" : src ? "Original upload" : "No media"}
          </span>
          <span className="rounded-full bg-secondary px-2.5 py-1 text-muted-foreground">Feed #{videoId}</span>
        </div>
        <span className="text-xs text-muted-foreground">
          {date} • {timestamp}
        </span>
      </div>

      {src ? (
        <div className="bg-black">
          <div ref={frameRef} className="relative aspect-video w-full bg-black">
            <video
              key={src}
              ref={resolvedRef}
              src={src}
              controls
              playsInline
              preload="metadata"
              className="aspect-video w-full bg-black"
            />
            {showAnyOverlay && (
              <svg
                aria-hidden="true"
                viewBox={`0 0 ${overlayWidth} ${overlayHeight}`}
                preserveAspectRatio="none"
                className="pointer-events-none absolute inset-0 z-10 h-full w-full"
              >
                {showROI && roiPolygons.map((polygon, index) => (
                  <polygon
                    key={`roi-${index}`}
                    points={toOverlayPointString(polygon, overlayWidth, overlayHeight)}
                    fill="none"
                    stroke="#39FF14"
                    strokeWidth={overlayStrokeWidth}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                    style={{ filter: "drop-shadow(0 0 6px rgba(57, 255, 20, 0.9))" }}
                  />
                ))}
                {showEntryExitPoints && directionalZoneOverlays.map((zone) => (
                  <g key={zone.key}>
                    <polygon
                      points={zone.points}
                      fill="none"
                      stroke={zone.stroke}
                      strokeWidth={overlayStrokeWidth}
                      strokeLinejoin="round"
                      strokeLinecap="round"
                      style={{ filter: `drop-shadow(0 0 6px ${zone.stroke}99)` }}
                    />
                    <text
                      x={zone.centroid.x}
                      y={zone.centroid.y}
                      textAnchor="middle"
                      dominantBaseline="middle"
                      fontSize={labelFontSize}
                      fontWeight={700}
                      fill={zone.stroke}
                      stroke="rgba(0,0,0,0.9)"
                      strokeWidth={labelFontSize * 0.18}
                      paintOrder="stroke"
                    >
                      {zone.label}
                    </text>
                  </g>
                ))}
              </svg>
            )}
          </div>
        </div>
      ) : (
        <div className="flex aspect-video items-center justify-center gap-3 bg-secondary px-6 text-center text-muted-foreground">
          <AlertCircle className="h-5 w-5 shrink-0" />
          <p className="text-sm">No uploaded media is available for this video yet.</p>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 px-4 py-3 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-secondary px-2.5 py-1">
          <MapPin className="h-3.5 w-3.5" />
          {location}
        </span>
        <span className="inline-flex items-center gap-1.5 rounded-full bg-secondary px-2.5 py-1">
          <Users className="h-3.5 w-3.5" />
          {pedestrianCount} pedestrians
        </span>
      </div>
    </div>
  )
}
