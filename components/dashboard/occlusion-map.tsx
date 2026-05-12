"use client"

import { useEffect, useMemo, useRef } from "react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import type { LatLngBoundsExpression } from "leaflet"
import type { PTSILocation, PTSIMapResponse, PTSIState } from "@/lib/api"
import { Clock, Info } from "lucide-react"

interface OcclusionMapProps {
  hourFilter: string
  onHourFilterChange: (hour: string) => void
  data?: PTSIMapResponse | null
  loading?: boolean
}

function formatHourLabel(hour: string) {
  const [rawHours, rawMinutes] = hour.split(":")
  const hours = Number(rawHours)
  const minutes = Number(rawMinutes ?? 0)
  const suffix = hours >= 12 ? "PM" : "AM"
  const displayHours = ((hours + 11) % 12) + 1
  return `${displayHours}:${String(minutes).padStart(2, "0")} ${suffix}`
}

function resolveSelectedHourData(location: PTSILocation, hourFilter: string) {
  if (hourFilter === "all") {
    return {
      score: location.score ?? null,
      mode: location.mode ?? null,
      averagePedestrians: location.averagePedestrians ?? null,
      uniquePedestrians: location.uniquePedestrians ?? null,
      occlusionMix: location.occlusionMix ?? null,
      los: location.los ?? null,
      losDescription: location.losDescription ?? null,
    }
  }

  const hourData = location.hourlyScores.find((item) => item.hour === hourFilter)
  return {
    score: hourData?.score ?? null,
    mode: hourData?.mode ?? location.mode ?? null,
    averagePedestrians: hourData?.averagePedestrians ?? null,
    uniquePedestrians: hourData?.uniquePedestrians ?? null,
    occlusionMix: hourData?.occlusionMix ?? null,
    los: hourData?.los ?? null,
    losDescription: hourData?.losDescription ?? null,
  }
}

function resolveStateFromLos(los: PTSILocation["los"]): PTSIState {
  if (los === "A" || los === "B" || los === "C") {
    return "clear"
  }
  if (los === "D" || los === "E") {
    return "moderate"
  }
  if (los === "F") {
    return "severe"
  }
  return "no-data"
}

function formatSeverityCategory(state: PTSIState) {
  if (state === "clear") {
    return "Light severity"
  }
  if (state === "moderate") {
    return "Moderate severity"
  }
  if (state === "severe") {
    return "High severity"
  }
  return null
}

function resolveState(location: PTSILocation, hourFilter: string): PTSIState {
  if (hourFilter === "all") {
    return location.state
  }
  if (!location.hasFootage) {
    return "no-footage"
  }

  const detail = resolveSelectedHourData(location, hourFilter)
  if (detail.los == null) {
    return "no-data"
  }
  return resolveStateFromLos(detail.los)
}

function getSeverityVisual(state: PTSIState) {
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

function formatLosLabel(los: PTSILocation["los"]) {
  return los ? `LOS ${los}` : "LOS —"
}

function getStateLabel(state: PTSIState, los: PTSILocation["los"]) {
  switch (state) {
    case "clear":
      return formatLosLabel(los)
    case "moderate":
      return formatLosLabel(los)
    case "severe":
      return formatLosLabel(los)
    case "no-data":
      return "No LOS data"
    default:
      return "No footage"
  }
}

function getStateSummary(state: PTSIState, detail: ReturnType<typeof resolveSelectedHourData>) {
  if (state === "no-footage") {
    return "No footage in selected range"
  }

  if (state === "no-data") {
    return "Footage available, but no ROI-qualified pedestrian traffic data yet"
  }

  return `${formatSeverityCategory(state)} · ${formatLosLabel(detail.los)} · ${detail.losDescription ?? "FHWA/HCM walkway interpretation unavailable"} · PTSI ${detail.score?.toFixed(1) ?? "0.0"}%`
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

const LOS_GRADES = ["A", "B", "C", "D", "E", "F"] as const

const LOS_RANKS = LOS_GRADES.reduce<Record<string, number>>((accumulator, grade, index) => {
  accumulator[grade] = index
  return accumulator
}, {})

function colorForLos(los: string | null): string {
  if (los === "A") return "#22C55E"
  if (los === "B") return "#84CC16"
  if (los === "C") return "#EAB308"
  if (los === "D") return "#F97316"
  if (los === "E") return "#EF4444"
  if (los === "F") return "#B91C1C"
  return "#94A3B8"
}

function getHeatmapStyle(los: string | null) {
  const rank = los ? (LOS_RANKS[los] ?? 0) : 0
  const severity = (rank + 1) / LOS_GRADES.length
  return {
    outerRadiusMeters: 50 + severity * 95,
    innerRadiusMeters: 22 + severity * 48,
    outerOpacity: 0.08 + severity * 0.24,
    innerOpacity: 0.18 + severity * 0.42,
  }
}

function createTrianglePinHtml(fillColor: string): string {
  const sizePx = 26
  return `<div style="width:${sizePx}px; height:${sizePx}px; filter:drop-shadow(0 1px 2px rgba(0,0,0,0.45));">
    <svg viewBox="0 0 24 24" width="${sizePx}" height="${sizePx}" xmlns="http://www.w3.org/2000/svg">
      <polygon points="12,2 22,21 2,21" fill="${fillColor}" stroke="#ffffff" stroke-width="1.8" stroke-linejoin="round" />
    </svg>
  </div>`
}

function buildPopupContent(location: PTSILocation, detail: ReturnType<typeof resolveSelectedHourData>, hourFilter: string, state: PTSIState): HTMLDivElement {
  const wrapper = document.createElement("div")
  wrapper.style.fontFamily = "Inter, system-ui, sans-serif"
  wrapper.style.minWidth = "200px"

  const title = document.createElement("div")
  title.style.fontWeight = "700"
  title.style.fontSize = "13px"
  title.style.marginBottom = "4px"
  title.textContent = location.name
  wrapper.appendChild(title)

  const subtitle = document.createElement("div")
  subtitle.style.fontSize = "11px"
  subtitle.style.color = "#94A3B8"
  subtitle.style.marginBottom = "6px"
  subtitle.textContent = "Pedestrian walkway — PTSI"
  wrapper.appendChild(subtitle)

  const appendRow = (label: string, value: string) => {
    const row = document.createElement("div")
    row.style.fontSize = "12px"
    row.style.marginBottom = "2px"
    const strong = document.createElement("strong")
    strong.textContent = `${label}: `
    row.appendChild(strong)
    row.appendChild(document.createTextNode(value))
    wrapper.appendChild(row)
  }

  const losValue = detail.los ? `LOS ${detail.los}` : "—"
  appendRow("LOS", losValue)
  appendRow("Score", detail.score?.toFixed(1) ?? "—")
  appendRow("State", formatSeverityCategory(state) ?? "—")
  appendRow("Hour", hourFilter === "all" ? "All hours" : formatHourLabel(hourFilter))

  const coords = document.createElement("div")
  coords.style.marginTop = "6px"
  coords.style.color = "#64748B"
  coords.style.fontSize = "11px"
  coords.textContent = `${location.latitude.toFixed(6)}°, ${location.longitude.toFixed(6)}°`
  wrapper.appendChild(coords)

  return wrapper
}

const OSM_TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
const OSM_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

export function OcclusionMap({ hourFilter, onHourFilterChange, data, loading = false }: OcclusionMapProps) {
  const locations = data?.locations ?? []
  const availableHours = useMemo(() => Array.from(new Set(data?.availableHours ?? [])), [data?.availableHours])
  const mapHostRef = useRef<HTMLDivElement | null>(null)

  const markerData = useMemo(() => {
    return locations.map((location) => {
      const detail = resolveSelectedHourData(location, hourFilter)
      const state = resolveState(location, hourFilter)
      const los = detail.los
      const color = colorForLos(los)
      return { location, detail, state, los, color }
    })
  }, [locations, hourFilter])

  useEffect(() => {
    const host = mapHostRef.current
    if (!host || markerData.length === 0) return

    let cancelled = false
    let resizeObserver: ResizeObserver | null = null
    let delayedInvalidateTimer: ReturnType<typeof setTimeout> | null = null

    void (async () => {
      const L = await import("leaflet")
      if (cancelled || !host) return

      const map = L.map(host, {
        minZoom: 15,
        maxZoom: 20,
        zoomControl: true,
        scrollWheelZoom: true,
      })

      L.tileLayer(OSM_TILE_URL, {
        maxZoom: 20,
        attribution: OSM_ATTRIBUTION,
      }).addTo(map)

      const bounds: LatLngBoundsExpression = markerData.map((m) => [m.location.latitude, m.location.longitude] as [number, number])
      map.fitBounds(bounds, { padding: [32, 32], maxZoom: 18 })

      const safelyInvalidate = () => {
        if (!cancelled) map.invalidateSize()
      }
      requestAnimationFrame(safelyInvalidate)
      delayedInvalidateTimer = setTimeout(safelyInvalidate, 180)

      resizeObserver = new ResizeObserver(() => {
        if (!cancelled) map.invalidateSize()
      })
      resizeObserver.observe(host)

      // Place markers for each pedestrian location
      markerData.forEach(({ location, detail, state, los, color }) => {
        const lat = location.latitude
        const lng = location.longitude
        const heatmap = getHeatmapStyle(los)

        // Outer heatmap glow circle
        L.circle([lat, lng], {
          radius: heatmap.outerRadiusMeters,
          stroke: false,
          fillColor: color,
          fillOpacity: heatmap.outerOpacity,
          interactive: false,
        }).addTo(map)

        // Inner heatmap circle
        L.circle([lat, lng], {
          radius: heatmap.innerRadiusMeters,
          stroke: false,
          fillColor: color,
          fillOpacity: heatmap.innerOpacity,
          interactive: false,
        }).addTo(map)

        // Triangle pedestrian pin
        const icon = L.divIcon({
          className: "",
          iconSize: [26, 28],
          iconAnchor: [13, 26],
          popupAnchor: [0, -27],
          html: createTrianglePinHtml(color),
        })

        const marker = L.marker([lat, lng], { icon }).addTo(map)
        const popupContent = buildPopupContent(location, detail, hourFilter, state)
        marker.bindPopup(popupContent)

        marker.bindTooltip(los ? `LOS ${los}` : location.name, {
          direction: "top",
          offset: [0, -8],
          opacity: 0.92,
        })
        marker.on("mouseover", () => marker.openTooltip())
        marker.on("click", () => {
          map.flyTo([lat, lng], Math.max(map.getZoom(), 18), { duration: 0.45 })
        })
      })

      // Only remove when all is said and done
      // We don't clean up on unmount here because the ref-based pattern is safe
      // with React strict mode in development
    })()

    return () => {
      cancelled = true
      if (delayedInvalidateTimer !== null) clearTimeout(delayedInvalidateTimer)
      resizeObserver?.disconnect()
    }
  }, [markerData, hourFilter])

  return (
    <div className="rounded-3xl border border-border bg-card p-6 shadow-elevated">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-foreground">Pedestrian Traffic Severity Index Map</h3>
            <Tooltip>
              <TooltipTrigger asChild>
                <button type="button" aria-label="Explain the Pedestrian Traffic Severity Index" className="rounded-full text-muted-foreground transition-colors hover:text-foreground">
                  <Info className="h-4 w-4" />
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom" sideOffset={8} className="max-w-72 rounded-xl px-3 py-2 text-xs">
                PTSI uses the 90th percentile of 100 × (0.85 × congestion + 0.15 × occlusion). Strict FHWA mode derives LOS from walkable area and space-per-pedestrian thresholds; FHWA-inspired provisional interpretation keeps the internal PTSI score and maps score bands into LOS labels for the UI.
              </TooltipContent>
            </Tooltip>
          </div>
          <p className="text-sm text-muted-foreground">LOS-first interpretation using ROI-qualified pedestrian tracks, with PTSI shown as supporting detail</p>
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

      <div className="relative">
        {loading ? (
          <div className="flex h-72 items-center justify-center rounded-2xl border border-border/60 bg-secondary/20 text-sm text-muted-foreground">
            <svg className="mr-2 h-4 w-4 animate-spin" viewBox="0 0 24 24" fill="none">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Calculating PTSI map...
          </div>
        ) : locations.length === 0 ? (
          <div className="flex h-72 items-center justify-center rounded-2xl border border-border/60 bg-secondary/20 text-sm text-muted-foreground">
            No pedestrian location data available for the selected date.
          </div>
        ) : (
          <div className="relative overflow-hidden rounded-xl border border-border/60">
            <div ref={mapHostRef} className="h-72 w-full" />
            {/* LOS heat scale legend overlay */}
            <div className="pointer-events-none absolute bottom-2 right-2 rounded-md border border-slate-300/60 bg-white/90 px-2 py-1 text-[10px] text-slate-700 shadow-sm backdrop-blur">
              <div className="mb-1 font-semibold">LOS Heat</div>
              <div className="flex items-center gap-1">
                {LOS_GRADES.map((grade) => (
                  <span
                    key={grade}
                    className="inline-flex h-4 w-4 items-center justify-center rounded-full text-[9px] font-bold text-white"
                    style={{ backgroundColor: colorForLos(grade) }}
                    title={`LOS ${grade}`}
                  >
                    {grade}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="mt-4 rounded-2xl border border-border/60 bg-secondary/40 p-3 text-xs text-muted-foreground">
        Triangle pins show location-level PTSI severity on an OpenStreetMap campus view. Pin color follows the LOS heat scale below. Click any pin for details including LOS grade, mode, pedestrian counts, and peak/off-peak hour summaries. Pin positions match walkway coordinates from the main landing-page map.
      </div>

      {locations.length > 0 && (
        <div className="mt-4 rounded-2xl border border-border/60 bg-secondary/20 p-3">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold text-foreground">Location key</p>
              <p className="text-[11px] text-muted-foreground">Full location names and details for the markers shown on the map above.</p>
            </div>
            <span className="rounded-full border border-border/70 bg-background/70 px-2.5 py-1 text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
              {locations.length} locations
            </span>
          </div>

          <div className="grid max-h-56 grid-cols-1 gap-2 overflow-y-auto pr-1 sm:grid-cols-2">
            {locations.map((location) => {
              const detail = resolveSelectedHourData(location, hourFilter)
              const state = resolveState(location, hourFilter)
              const visual = getSeverityVisual(state)

              return (
                <div
                  key={`key-${location.id}`}
                  className="flex w-full items-start gap-3 rounded-xl border border-border/60 bg-background/45 px-3 py-2 text-left"
                >
                  <span
                    className="mt-0.5 inline-flex max-w-[5.75rem] items-center justify-center rounded-full border px-2 py-1 text-[10px] font-semibold leading-none truncate whitespace-nowrap"
                    style={{
                      color: visual.bg,
                      borderColor: `${visual.border}`,
                      backgroundColor: `${visual.bg}18`,
                    }}
                  >
                    {location.name}
                  </span>

                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="text-xs font-medium leading-tight text-foreground">{location.name}</p>
                      <span className="rounded-full bg-secondary px-2 py-0.5 text-[10px] font-medium text-muted-foreground">
                        {getStateLabel(state, detail.los)}
                      </span>
                    </div>
                    <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">{getStateSummary(state, detail)}</p>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      <div className="mt-4 grid grid-cols-1 gap-3 border-t border-border pt-4 sm:grid-cols-2">
        <LegendItem color="#FACC15" label="Light severity (LOS A–C)" detail="Very high space to manageable pedestrian interaction" />
        <LegendItem color="#F97316" label="Moderate severity (LOS D–E)" detail="Limited space to crowded conditions with frequent interference" />
        <LegendItem color="#EF4444" label="High severity (LOS F)" detail="Severely congested conditions and breakdown of smooth movement" />
        <LegendItem color="#64748B" label="Neutral marker" detail="No footage or no ROI-qualified PTSI data" />
      </div>
    </div>
  )
}
