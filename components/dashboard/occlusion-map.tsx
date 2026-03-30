"use client"

import { Fragment, useEffect, useMemo, useRef, useState, type CSSProperties } from "react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { OcclusionLocation, OcclusionMapResponse, OcclusionState } from "@/lib/api"
import { Clock, Info, Loader2 } from "lucide-react"

interface OcclusionMapProps {
  hourFilter: string
  onHourFilterChange: (hour: string) => void
  data?: OcclusionMapResponse | null
  loading?: boolean
}

interface MapPoint {
  x: number
  y: number
}

interface MapDimensions {
  width: number
  height: number
}

interface Rect {
  left: number
  top: number
  width: number
  height: number
}

type LabelDirection = "top" | "bottom" | "left" | "right" | "top-left" | "top-right" | "bottom-left" | "bottom-right"
type PlottedLocation = OcclusionLocation & { shortLabel: string; position: MapPoint }

const DEFAULT_MAP_DIMENSIONS: MapDimensions = { width: 720, height: 288 }
const MAP_LABEL_MIN_WIDTH = 52
const MAP_LABEL_MAX_WIDTH = 104
const MAP_LABEL_HEIGHT = 24
const MAP_LABEL_HORIZONTAL_PADDING = 18
const MAP_LABEL_COLLISION_PADDING = 6
const MAP_LABEL_BOUNDS_PADDING = 8
const MAP_PIN_EXCLUSION_RADIUS = 18
const MAP_LABEL_GAPS = [10, 18, 28]
const LABEL_DIRECTIONS: LabelDirection[] = ["top", "top-right", "top-left", "right", "left", "bottom", "bottom-right", "bottom-left"]

function formatHourLabel(hour: string) {
  const [rawHours, rawMinutes] = hour.split(":")
  const hours = Number(rawHours)
  const minutes = Number(rawMinutes ?? 0)
  const suffix = hours >= 12 ? "PM" : "AM"
  const displayHours = ((hours + 11) % 12) + 1
  return `${displayHours}:${String(minutes).padStart(2, "0")} ${suffix}`
}

function resolveHourlyScore(location: OcclusionLocation, hourFilter: string) {
  if (hourFilter === "all") {
    return location.score ?? null
  }

  return location.hourlyScores.find((item) => item.hour === hourFilter)?.score ?? null
}

function resolveState(location: OcclusionLocation, hourFilter: string): OcclusionState {
  if (hourFilter === "all") {
    return location.state
  }
  if (!location.hasFootage) {
    return "no-footage"
  }

  const score = resolveHourlyScore(location, hourFilter)
  if (score === null) {
    return "no-data"
  }
  if (score <= 32) {
    return "clear"
  }
  if (score <= 65) {
    return "moderate"
  }
  return "severe"
}

function getSeverityVisual(state: OcclusionState) {
  switch (state) {
    case "clear":
      return { bg: "#FACC15", glow: "rgba(250, 204, 21, 0.38)", shadow: "0 0 22px 8px rgba(250, 204, 21, 0.28)", border: "rgba(254, 249, 195, 0.95)", size: 38 }
    case "moderate":
      return { bg: "#F97316", glow: "rgba(249, 115, 22, 0.5)", shadow: "0 0 26px 10px rgba(249, 115, 22, 0.34)", border: "rgba(255, 237, 213, 0.95)", size: 50 }
    case "severe":
      return { bg: "#EF4444", glow: "rgba(239, 68, 68, 0.58)", shadow: "0 0 30px 12px rgba(239, 68, 68, 0.4)", border: "rgba(254, 226, 226, 0.95)", size: 62 }
    case "no-data":
      return { bg: "#94A3B8", glow: "rgba(148, 163, 184, 0)", shadow: "none", border: "rgba(226, 232, 240, 0.9)", size: 0 }
    default:
      return { bg: "#52525B", glow: "rgba(82, 82, 91, 0)", shadow: "none", border: "rgba(212, 212, 216, 0.85)", size: 0 }
  }
}

function describeState(state: OcclusionState, score: number | null) {
  if (state === "no-footage") {
    return "No footage for this location in the selected date range"
  }
  if (state === "no-data") {
    return "Footage exists, but no occlusion-class detections are available yet"
  }

  const label = state === "clear" ? "Soft Yellow Glow" : state === "moderate" ? "Vibrant Orange Glow" : "Intense Red Glow"
  return `${label} · OWDI ${score?.toFixed(1) ?? "0.0"}%`
}

function getTooltipPlacement(x: number, y: number): { style: CSSProperties; className: string } {
  const horizontalAnchor = x <= 28 ? "left" : x >= 72 ? "right" : "center"
  const verticalAnchor = y <= 36 ? "below" : "above"
  const style: CSSProperties = {}

  if (horizontalAnchor === "left") {
    style.left = `calc(${x}% + 0.85rem)`
  } else if (horizontalAnchor === "right") {
    style.right = `calc(${100 - x}% + 0.85rem)`
  } else {
    style.left = `${x}%`
  }

  if (verticalAnchor === "below") {
    style.top = `calc(${y}% + 0.85rem)`
  } else {
    style.bottom = `calc(${100 - y}% + 0.85rem)`
  }

  return {
    style,
    className: horizontalAnchor === "center" ? "-translate-x-1/2" : "",
  }
}

function getCompactLocationLabel(name: string) {
  const compactLabel = name
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .join(" ")

  if (compactLabel.length <= 14) {
    return compactLabel
  }

  return `${compactLabel.slice(0, 13).trimEnd()}…`
}

function clamp(value: number, min: number, max: number) {
  if (max < min) {
    return min
  }

  return Math.min(Math.max(value, min), max)
}

function estimateLabelWidth(label: string) {
  return Math.min(MAP_LABEL_MAX_WIDTH, Math.max(MAP_LABEL_MIN_WIDTH, label.length * 6.4 + MAP_LABEL_HORIZONTAL_PADDING))
}

function createLabelRect(direction: LabelDirection, point: MapPoint, width: number, height: number, gap: number): Rect {
  switch (direction) {
    case "top":
      return { left: point.x - width / 2, top: point.y - gap - height, width, height }
    case "bottom":
      return { left: point.x - width / 2, top: point.y + gap, width, height }
    case "left":
      return { left: point.x - gap - width, top: point.y - height / 2, width, height }
    case "right":
      return { left: point.x + gap, top: point.y - height / 2, width, height }
    case "top-left":
      return { left: point.x - gap - width, top: point.y - gap - height, width, height }
    case "top-right":
      return { left: point.x + gap, top: point.y - gap - height, width, height }
    case "bottom-left":
      return { left: point.x - gap - width, top: point.y + gap, width, height }
    default:
      return { left: point.x + gap, top: point.y + gap, width, height }
  }
}

function clampRectToBounds(rect: Rect, mapDimensions: MapDimensions) {
  const maxLeft = Math.max(MAP_LABEL_BOUNDS_PADDING, mapDimensions.width - MAP_LABEL_BOUNDS_PADDING - rect.width)
  const maxTop = Math.max(MAP_LABEL_BOUNDS_PADDING, mapDimensions.height - MAP_LABEL_BOUNDS_PADDING - rect.height)
  const left = clamp(rect.left, MAP_LABEL_BOUNDS_PADDING, maxLeft)
  const top = clamp(rect.top, MAP_LABEL_BOUNDS_PADDING, maxTop)

  return {
    rect: { ...rect, left, top },
    displacement: Math.abs(left - rect.left) + Math.abs(top - rect.top),
  }
}

function expandRect(rect: Rect, padding: number): Rect {
  return {
    left: rect.left - padding,
    top: rect.top - padding,
    width: rect.width + padding * 2,
    height: rect.height + padding * 2,
  }
}

function getRectIntersectionArea(a: Rect, b: Rect) {
  const overlapWidth = Math.max(0, Math.min(a.left + a.width, b.left + b.width) - Math.max(a.left, b.left))
  const overlapHeight = Math.max(0, Math.min(a.top + a.height, b.top + b.height) - Math.max(a.top, b.top))
  return overlapWidth * overlapHeight
}

function getCircleIntrusion(rect: Rect, x: number, y: number, radius: number) {
  const nearestX = clamp(x, rect.left, rect.left + rect.width)
  const nearestY = clamp(y, rect.top, rect.top + rect.height)
  const distance = Math.hypot(x - nearestX, y - nearestY)
  return Math.max(0, radius - distance)
}

function getMarkerDensity(target: { id: string; x: number; y: number }, markers: Array<{ id: string; x: number; y: number }>) {
  return markers.reduce((score, marker) => {
    if (marker.id === target.id) {
      return score
    }

    const distance = Math.hypot(target.x - marker.x, target.y - marker.y)
    if (distance >= 170) {
      return score
    }

    return score + (170 - distance) / 170
  }, 0)
}

function buildLabelPlacements(plottedLocations: PlottedLocation[], mapDimensions: MapDimensions) {
  const locationsWithPixels = plottedLocations.map((location) => ({
    ...location,
    x: (location.position.x / 100) * mapDimensions.width,
    y: (location.position.y / 100) * mapDimensions.height,
    labelWidth: estimateLabelWidth(location.shortLabel),
  }))
  const pinZones = locationsWithPixels.map((location) => ({ id: location.id, x: location.x, y: location.y }))
  const orderedLocations = [...locationsWithPixels].sort((left, right) => {
    const densityDelta = getMarkerDensity(right, pinZones) - getMarkerDensity(left, pinZones)
    if (Math.abs(densityDelta) > 0.01) {
      return densityDelta
    }

    return right.labelWidth - left.labelWidth
  })
  const placedRects = new Map<string, Rect>()

  orderedLocations.forEach((location) => {
    let bestPlacement: { rect: Rect; score: number } | null = null

    MAP_LABEL_GAPS.forEach((gap, gapIndex) => {
      LABEL_DIRECTIONS.forEach((direction, directionIndex) => {
        const rawRect = createLabelRect(direction, { x: location.x, y: location.y }, location.labelWidth, MAP_LABEL_HEIGHT, gap)
        const { rect, displacement } = clampRectToBounds(rawRect, mapDimensions)
        const collisionRect = expandRect(rect, MAP_LABEL_COLLISION_PADDING)
        const labelCenterX = rect.left + rect.width / 2
        const labelCenterY = rect.top + rect.height / 2
        let score = displacement * 30 + gapIndex * 4 + directionIndex * 0.6 + Math.hypot(labelCenterX - location.x, labelCenterY - location.y) * 0.18

        pinZones.forEach((pinZone) => {
          const intrusion = getCircleIntrusion(collisionRect, pinZone.x, pinZone.y, MAP_PIN_EXCLUSION_RADIUS)
          if (intrusion > 0) {
            score += (pinZone.id === location.id ? 6000 : 4500) + intrusion * 600
          }
        })

        placedRects.forEach((placedRect) => {
          const overlapArea = getRectIntersectionArea(collisionRect, expandRect(placedRect, MAP_LABEL_COLLISION_PADDING))
          if (overlapArea > 0) {
            score += 120000 + overlapArea * 40
          }
        })

        if (!bestPlacement || score < bestPlacement.score) {
          bestPlacement = { rect, score }
        }
      })
    })

    if (bestPlacement) {
      placedRects.set(location.id, (bestPlacement as { rect: Rect; score: number }).rect)
    }
  })

  return Object.fromEntries(
    Array.from(placedRects.entries()).map(([id, rect]) => [id, { left: rect.left, top: rect.top }]),
  ) as Record<string, { left: number; top: number }>
}

function getStateLabel(state: OcclusionState) {
  switch (state) {
    case "clear":
      return "Clear"
    case "moderate":
      return "Moderate"
    case "severe":
      return "Severe"
    case "no-data":
      return "No data"
    default:
      return "No footage"
  }
}

function getStateSummary(state: OcclusionState, score: number | null) {
  if (state === "no-footage") {
    return "No footage in selected range"
  }

  if (state === "no-data") {
    return "Footage available, no occlusion data yet"
  }

  return `OWDI ${score?.toFixed(1) ?? "0.0"}%`
}

function LegendItem({ color, label, detail }: { color: string; label: string; detail: string }) {
  return (
    <div className="flex items-start gap-3 rounded-2xl border border-border/60 bg-secondary/30 p-3">
      <div className="mt-0.5 h-3 w-3 rounded-full" style={{ backgroundColor: color, boxShadow: `0 0 12px ${color}` }} />
      <div>
        <p className="text-xs font-medium text-foreground">{label}</p>
        <p className="text-[11px] text-muted-foreground">{detail}</p>
      </div>
    </div>
  )
}

export function OcclusionMap({ hourFilter, onHourFilterChange, data, loading = false }: OcclusionMapProps) {
  const [hoveredLocationId, setHoveredLocationId] = useState<string | null>(null)
  const mapContainerRef = useRef<HTMLDivElement | null>(null)
  const [mapDimensions, setMapDimensions] = useState<MapDimensions>(DEFAULT_MAP_DIMENSIONS)
  const locations = data?.locations ?? []
  const availableHours = data?.availableHours ?? []
  const plottedLocations = useMemo<PlottedLocation[]>(() => {
    const minLat = locations.length > 0 ? Math.min(...locations.map((location) => location.latitude)) : 0
    const maxLat = locations.length > 0 ? Math.max(...locations.map((location) => location.latitude)) : 1
    const minLng = locations.length > 0 ? Math.min(...locations.map((location) => location.longitude)) : 0
    const maxLng = locations.length > 0 ? Math.max(...locations.map((location) => location.longitude)) : 1
    const getPosition = (lat: number, lng: number): MapPoint => {
      const latRange = maxLat - minLat || 0.01
      const lngRange = maxLng - minLng || 0.01
      return {
        x: ((lng - minLng) / lngRange) * 66 + 16,
        y: ((maxLat - lat) / latRange) * 42 + 12,
      }
    }

    return locations.map((location) => ({
      ...location,
      shortLabel: getCompactLocationLabel(location.name),
      position: getPosition(location.latitude, location.longitude),
    }))
  }, [locations])
  const labelPlacements = useMemo(() => buildLabelPlacements(plottedLocations, mapDimensions), [plottedLocations, mapDimensions])

  const routeOrder = ["gate-1-walkway", "edsa-sec-walk", "kostka-walk", "gate-3-walkway"]
  const routePoints = routeOrder
    .map((id) => plottedLocations.find((location) => location.id === id))
    .filter((location): location is (typeof plottedLocations)[number] => Boolean(location))
  const campusSpine = routePoints.map((location, index) => `${index === 0 ? "M" : "L"} ${location.position.x} ${location.position.y}`).join(" ")

  useEffect(() => {
    const mapElement = mapContainerRef.current
    if (!mapElement) {
      return
    }

    const updateDimensions = (width: number, height: number) => {
      const nextWidth = width > 0 ? width : DEFAULT_MAP_DIMENSIONS.width
      const nextHeight = height > 0 ? height : DEFAULT_MAP_DIMENSIONS.height

      setMapDimensions((current) => {
        if (Math.abs(current.width - nextWidth) < 1 && Math.abs(current.height - nextHeight) < 1) {
          return current
        }

        return {
          width: nextWidth,
          height: nextHeight,
        }
      })
    }

    const initialRect = mapElement.getBoundingClientRect()
    updateDimensions(initialRect.width, initialRect.height)

    if (typeof ResizeObserver === "undefined") {
      return
    }

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (!entry) {
        return
      }

      updateDimensions(entry.contentRect.width, entry.contentRect.height)
    })

    observer.observe(mapElement)

    return () => observer.disconnect()
  }, [])

  return (
    <div className="rounded-3xl border border-border bg-card p-6 shadow-elevated">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-foreground">Occlusion Severity Map</h3>
            <Tooltip>
              <TooltipTrigger asChild>
                <button type="button" aria-label="Explain sustained peak density" className="rounded-full text-muted-foreground transition-colors hover:text-foreground">
                  <Info className="h-4 w-4" />
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom" sideOffset={8} className="max-w-72 rounded-xl px-3 py-2 text-xs">
                Sustained Peak Density (90th Percentile) uses OWDI: (1×Light + 2×Moderate + 3×Heavy) ÷ (3×50) × 100. Locations with no footage stay neutral instead of glowing.
              </TooltipContent>
            </Tooltip>
          </div>
          <p className="text-sm text-muted-foreground">Sustained Peak Density (90th Percentile) by location</p>
        </div>

        <Select value={hourFilter} onValueChange={onHourFilterChange}>
          <SelectTrigger className="w-44 rounded-2xl border-border bg-secondary text-foreground">
            <Clock className="mr-2 h-4 w-4 text-muted-foreground" />
            <SelectValue placeholder="Hour" />
          </SelectTrigger>
          <SelectContent className="rounded-xl border-border bg-popover">
            <SelectItem value="all" className="rounded-lg text-foreground">All Hours</SelectItem>
            {availableHours.map((hour) => (
              <SelectItem key={hour} value={hour} className="rounded-lg text-foreground">
                {formatHourLabel(hour)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div ref={mapContainerRef} className="relative h-72 overflow-hidden rounded-2xl border border-white/5 bg-[#0f172a]">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(56,189,248,0.12),_transparent_34%),radial-gradient(circle_at_bottom_right,_rgba(34,197,94,0.12),_transparent_30%)]" />
        <svg viewBox="0 0 100 70" className="pointer-events-none absolute inset-0 h-full w-full opacity-70">
          <path d="M8 50 L18 16 L36 12 L56 16 L72 22 L88 34 L90 58 L14 60 Z" fill="rgba(15,23,42,0.9)" stroke="rgba(148,163,184,0.22)" strokeWidth="1.2" />
          <path d="M10 19 L30 12 L48 14" fill="none" stroke="rgba(59,130,246,0.28)" strokeWidth="1.2" strokeDasharray="4 3" />
          <path d="M18 30 L40 24 L56 30 L76 40" fill="none" stroke="rgba(148,163,184,0.18)" strokeWidth="1.2" strokeDasharray="4 3" />
          <rect x="22" y="18" width="14" height="10" rx="2" fill="rgba(30,41,59,0.75)" stroke="rgba(148,163,184,0.18)" />
          <rect x="40" y="20" width="13" height="9" rx="2" fill="rgba(30,41,59,0.7)" stroke="rgba(148,163,184,0.18)" />
          <rect x="60" y="26" width="12" height="8" rx="2" fill="rgba(30,41,59,0.7)" stroke="rgba(148,163,184,0.18)" />
          {campusSpine && <path d={campusSpine} fill="none" stroke="rgba(34,197,94,0.36)" strokeWidth="2.4" strokeLinecap="round" />}
        </svg>

        <div className="pointer-events-none absolute inset-0">
          {[0, 25, 50, 75, 100].map((percent) => (
            <div key={`h-${percent}`} className="absolute w-full border-t border-white/6" style={{ top: `${percent}%` }} />
          ))}
          {[0, 25, 50, 75, 100].map((percent) => (
            <div key={`v-${percent}`} className="absolute h-full border-l border-white/6" style={{ left: `${percent}%` }} />
          ))}
        </div>

        {loading ? (
          <div className="relative z-10 flex h-full items-center justify-center text-sm text-slate-300">
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Calculating OWDI map...
          </div>
        ) : (
          <>
            {plottedLocations.map((location) => {
              const pos = location.position
              const score = resolveHourlyScore(location, hourFilter)
              const state = resolveState(location, hourFilter)
              const visual = getSeverityVisual(state)
              const isGlowing = state === "clear" || state === "moderate" || state === "severe"
              const tooltipPlacement = getTooltipPlacement(pos.x, pos.y)
              const badgePlacement = labelPlacements[location.id]
              const isHovered = hoveredLocationId === location.id

              return (
                <Fragment key={location.id}>
                  {isGlowing && (
                    <div
                      className="pointer-events-none absolute animate-pulse rounded-full"
                      style={{
                        width: `${visual.size}px`,
                        height: `${visual.size}px`,
                        left: `${pos.x}%`,
                        top: `${pos.y}%`,
                        transform: "translate(-50%, -50%)",
                        background: `radial-gradient(circle, ${visual.glow} 0%, transparent 72%)`,
                        boxShadow: visual.shadow,
                      }}
                    />
                  )}

                  <button
                    type="button"
                    aria-label={`Show occlusion details for ${location.name}`}
                    className="absolute z-20 flex h-8 w-8 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full outline-none transition-transform hover:scale-105 focus-visible:scale-105 focus-visible:ring-2 focus-visible:ring-white/70"
                    style={{
                      left: `${pos.x}%`,
                      top: `${pos.y}%`,
                      opacity: state === "no-footage" ? 0.85 : 1,
                    }}
                    onMouseEnter={() => setHoveredLocationId(location.id)}
                    onMouseLeave={() => setHoveredLocationId((current) => (current === location.id ? null : current))}
                    onClick={() => setHoveredLocationId((current) => (current === location.id ? null : location.id))}
                    onFocus={() => setHoveredLocationId(location.id)}
                    onBlur={() => setHoveredLocationId((current) => (current === location.id ? null : current))}
                  >
                    <span
                      className="absolute h-6 w-6 rounded-full"
                      style={{
                        backgroundColor: `${visual.bg}20`,
                        border: `1px solid ${visual.border}`,
                      }}
                    />
                    <span
                      className="absolute h-3.5 w-3.5 rounded-full border-2 shadow-lg"
                      style={{
                        backgroundColor: visual.bg,
                        borderColor: visual.border,
                      }}
                    />
                  </button>

                  <div
                    className={`pointer-events-none absolute z-10 max-w-[6.5rem] rounded-full border border-white/15 bg-slate-950/95 px-2 py-1 text-[10px] font-semibold leading-none text-slate-50 shadow-sm backdrop-blur-sm transition-transform ${isHovered ? "scale-110" : "scale-100"}`}
                    style={{
                      left: `${badgePlacement?.left ?? (pos.x / 100) * mapDimensions.width}px`,
                      top: `${badgePlacement?.top ?? (pos.y / 100) * mapDimensions.height}px`,
                    }}
                  >
                    <span className="block truncate whitespace-nowrap">{location.shortLabel}</span>
                  </div>

                  <div
                    className={`pointer-events-none absolute z-30 w-44 max-w-[calc(100%-1.5rem)] rounded-xl border border-border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-elevated transition-opacity ${tooltipPlacement.className} ${isHovered ? "opacity-100" : "opacity-0"}`}
                    style={tooltipPlacement.style}
                  >
                    <p className="font-medium text-foreground">{location.name}</p>
                    <p className="mt-1 text-muted-foreground">{describeState(state, score)}</p>
                    <p className="mt-1 text-[11px] text-muted-foreground">
                      {hourFilter === "all" ? "Peak hour across the selected range" : `Hour window: ${formatHourLabel(hourFilter)}`}
                    </p>
                  </div>
                </Fragment>
              )
            })}

          </>
        )}
      </div>

      <div className="mt-4 rounded-2xl border border-border/60 bg-secondary/40 p-3 text-xs text-muted-foreground">
        Map labels are shortened to keep names readable as more locations are added. Neutral markers mean there is either no footage for that location on the selected date or no occlusion-class detections available for the chosen hour.
      </div>

      {plottedLocations.length > 0 && (
        <div className="mt-4 rounded-2xl border border-border/60 bg-secondary/20 p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold text-foreground">Location key</p>
              <p className="text-[11px] text-muted-foreground">Map labels use a shortened version of each location name. This list shows the full names.</p>
            </div>
            <span className="rounded-full border border-border/70 bg-background/70 px-2.5 py-1 text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              {plottedLocations.length} locations
            </span>
          </div>

          <div className="grid max-h-56 grid-cols-1 gap-2 overflow-y-auto pr-1 sm:grid-cols-2">
            {plottedLocations.map((location) => {
              const score = resolveHourlyScore(location, hourFilter)
              const state = resolveState(location, hourFilter)
              const visual = getSeverityVisual(state)
              const isHovered = hoveredLocationId === location.id

              return (
                <button
                  key={`key-${location.id}`}
                  type="button"
                  className={`flex w-full items-start gap-3 rounded-xl border px-3 py-2 text-left transition-colors ${isHovered ? "border-primary/40 bg-background/85 shadow-sm" : "border-border/60 bg-background/45 hover:bg-background/75"}`}
                  onMouseEnter={() => setHoveredLocationId(location.id)}
                  onMouseLeave={() => setHoveredLocationId((current) => (current === location.id ? null : current))}
                  onFocus={() => setHoveredLocationId(location.id)}
                  onBlur={() => setHoveredLocationId((current) => (current === location.id ? null : current))}
                  onClick={() => setHoveredLocationId((current) => (current === location.id ? null : location.id))}
                >
                  <span
                    className="mt-0.5 inline-flex max-w-[5.75rem] items-center justify-center rounded-full border px-2 py-1 text-[10px] font-semibold leading-none"
                    style={{
                      color: visual.bg,
                      borderColor: `${visual.border}`,
                      backgroundColor: `${visual.bg}18`,
                    }}
                  >
                    <span className="block truncate whitespace-nowrap">{location.shortLabel}</span>
                  </span>

                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="text-xs font-medium leading-tight text-foreground">{location.name}</p>
                      <span className="rounded-full bg-secondary px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                        {getStateLabel(state)}
                      </span>
                    </div>
                    <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">{getStateSummary(state, score)}</p>
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      )}

      <div className="mt-4 grid grid-cols-1 gap-3 border-t border-border pt-4 sm:grid-cols-2">
        <LegendItem color="#FACC15" label="Soft Yellow Glow" detail="0–32% · clear flow and normal spacing" />
        <LegendItem color="#F97316" label="Vibrant Orange Glow" detail="33–65% · moderate crowding and dense clusters" />
        <LegendItem color="#EF4444" label="Intense Red Glow" detail="66–100% · severe bottleneck and packed heavy occlusion" />
        <LegendItem color="#64748B" label="Neutral Marker" detail="No footage or no occlusion-class data" />
      </div>
    </div>
  )
}
