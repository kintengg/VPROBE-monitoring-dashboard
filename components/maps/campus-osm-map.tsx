"use client"

import { useEffect, useMemo, useRef } from "react"
import type { LatLngBoundsExpression, Map as LeafletMap } from "leaflet"
import type { PTSIMapResponse, PTSILocation } from "@/lib/api"

const CAMPUS_MAP_BOUNDS = {
  north: 14.64807,
  east: 121.08418,
  south: 14.63155,
  west: 121.07088,
} as const

const LANDMARKS = [
  { id: "gate-2", name: "Gate 2", lat: 14.635825, lng: 121.074719 },
  { id: "gate-2-9", name: "Gate 2.9", lat: 14.640421, lng: 121.074759 },
  { id: "gate-3", name: "Gate 3", lat: 14.640681, lng: 121.075508 },
  { id: "gate-3-2", name: "Gate 3.2", lat: 14.640904, lng: 121.074872 },
  { id: "gate-3-5", name: "Gate 3.5", lat: 14.64119, lng: 121.07477 },
] as const

function formatDateLabel(dateKey: string | null | undefined) {
  if (!dateKey) {
    return "No date selected"
  }

  const parsed = new Date(`${dateKey}T00:00:00`)
  if (Number.isNaN(parsed.getTime())) {
    return dateKey
  }

  return parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  })
}

interface CampusOsmMapProps {
  selectedDate?: string | null
  occlusionData?: PTSIMapResponse | null
  focusTime?: string
  timeRange?: string
  startTime?: string
  zoomLevel?: number
  selectedLocationId?: string
  showLosDetails?: boolean
  className?: string
}

const LOS_GRADES = ["A", "B", "C", "D", "E", "F"] as const

const LOS_RANKS = LOS_GRADES.reduce<Record<string, number>>((accumulator, grade, index) => {
  accumulator[grade] = index
  return accumulator
}, {})

const LOS_BY_NUMERIC_RANK: Record<number, string> = {
  1: "A",
  2: "B",
  3: "C",
  4: "D",
  5: "E",
  6: "F",
}

const TIME_RANGE_MINUTES: Record<string, number> = {
  "30m": 30,
  "1h": 60,
  "2h": 120,
  "3h": 180,
  "4h": 240,
  "6h": 360,
  "12h": 720,
  "whole-day": 24 * 60,
}

function formatTimeRangeLabel(timeRange: string | null | undefined) {
  const labels: Record<string, string> = {
    "30m": "30 minutes",
    "1h": "1 hour",
    "2h": "2 hours",
    "3h": "3 hours",
    "4h": "4 hours",
    "6h": "6 hours",
    "12h": "12 hours",
    "whole-day": "Whole day",
  }

  if (!timeRange) {
    return "Selected range"
  }

  return labels[timeRange] ?? timeRange
}

function normalizeLosGrade(value: unknown): string | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    const rankedGrade = LOS_BY_NUMERIC_RANK[Math.round(value)]
    return rankedGrade ?? null
  }

  const rawValue = String(value ?? "").trim()
  if (!rawValue) {
    return null
  }

  const normalized = rawValue.toUpperCase().replace(/^LOS\s*-?\s*/i, "")
  if (normalized in LOS_RANKS) {
    return normalized
  }

  const numericRank = Number.parseInt(normalized, 10)
  if (Number.isFinite(numericRank)) {
    return LOS_BY_NUMERIC_RANK[numericRank] ?? null
  }

  return null
}

function deriveLosFromScore(score: unknown): string | null {
  const scoreValue = typeof score === "number"
    ? score
    : typeof score === "string"
      ? Number.parseFloat(score)
      : Number.NaN

  if (!Number.isFinite(scoreValue)) {
    return null
  }

  if (scoreValue < 15) return "A"
  if (scoreValue < 33) return "B"
  if (scoreValue < 50) return "C"
  if (scoreValue < 66) return "D"
  if (scoreValue < 85) return "E"
  return "F"
}

function deriveLosFromDescription(description: unknown): string | null {
  const normalized = String(description ?? "").trim().toLowerCase()
  if (!normalized) {
    return null
  }

  if (normalized.includes("very high pedestrian space")) return "A"
  if (normalized.includes("high pedestrian space")) return "B"
  if (normalized.includes("moderate pedestrian space")) return "C"
  if (normalized.includes("limited pedestrian space")) return "D"
  if (normalized.includes("very limited pedestrian space")) return "E"
  if (normalized.includes("extremely limited pedestrian space")) return "F"
  return null
}

function resolveLosGrade(los: unknown, score: unknown, losDescription?: unknown): string | null {
  return normalizeLosGrade(los) ?? deriveLosFromScore(score) ?? deriveLosFromDescription(losDescription)
}

function extractHourOfDay(rawHour: string | null | undefined): number | null {
  if (!rawHour) {
    return null
  }

  const normalized = rawHour.trim().toUpperCase()
  const twentyFourHourMatch = normalized.match(/^([01]?\d|2[0-3])(?::([0-5]\d))?$/)
  if (twentyFourHourMatch) {
    const hourValue = Number.parseInt(twentyFourHourMatch[1] ?? "", 10)
    return Number.isFinite(hourValue) ? hourValue : null
  }

  const twelveHourMatch = normalized.match(/^([1-9]|1[0-2])(?::([0-5]\d))?\s*(AM|PM)$/)
  if (!twelveHourMatch) {
    return null
  }

  const hourValue = Number.parseInt(twelveHourMatch[1] ?? "", 10)
  if (!Number.isFinite(hourValue)) {
    return null
  }

  const period = twelveHourMatch[3]
  const twelveHour = hourValue % 12
  return period === "PM" ? twelveHour + 12 : twelveHour
}

function parseClockMinutes(value: string | null | undefined): number | null {
  const rawValue = String(value ?? "").trim()
  if (!rawValue) {
    return null
  }

  const normalized = rawValue.toUpperCase()
  const twentyFourHourMatch = normalized.match(/^([01]?\d|2[0-3]):([0-5]\d)$/)
  if (twentyFourHourMatch) {
    const hours = Number.parseInt(twentyFourHourMatch[1] ?? "", 10)
    const minutes = Number.parseInt(twentyFourHourMatch[2] ?? "", 10)
    if (!Number.isFinite(hours) || !Number.isFinite(minutes)) {
      return null
    }

    return (hours * 60) + minutes
  }

  const twelveHourMatch = normalized.match(/^([1-9]|1[0-2]):([0-5]\d)\s*(AM|PM)$/)
  if (!twelveHourMatch) {
    return null
  }

  const hourValue = Number.parseInt(twelveHourMatch[1] ?? "", 10)
  const minuteValue = Number.parseInt(twelveHourMatch[2] ?? "", 10)
  if (!Number.isFinite(hourValue) || !Number.isFinite(minuteValue)) {
    return null
  }

  const period = twelveHourMatch[3]
  const twelveHour = hourValue % 12
  const hours24 = period === "PM" ? twelveHour + 12 : twelveHour
  return (hours24 * 60) + minuteValue
}

function normalizeMinutesOfDay(value: number) {
  return ((value % 1440) + 1440) % 1440
}

function durationFromTimeRange(timeRange?: string) {
  if (!timeRange) {
    return null
  }

  return TIME_RANGE_MINUTES[timeRange] ?? null
}

function buildMarkerPopupContent(markerData: {
  name: string
  los: string
  sourceLabel: string
  lat: number
  lng: number
}, dateLabel: string, mapContextLabel: string) {
  const wrapper = document.createElement("div")
  wrapper.style.fontFamily = "Inter, system-ui, sans-serif"
  wrapper.style.minWidth = "160px"

  const title = document.createElement("div")
  title.style.fontWeight = "700"
  title.style.marginBottom = "4px"
  title.textContent = markerData.name
  wrapper.appendChild(title)

  const appendKeyValueRow = (label: string, value: string) => {
    const row = document.createElement("div")
    const strong = document.createElement("strong")
    strong.textContent = `${label}:`
    row.append(strong, ` ${value}`)
    wrapper.appendChild(row)
  }

  appendKeyValueRow("Date", dateLabel)
  appendKeyValueRow("Time focus", mapContextLabel)
  appendKeyValueRow("LOS", markerData.los)
  appendKeyValueRow("Source", markerData.sourceLabel)

  const coordinates = document.createElement("div")
  coordinates.style.marginTop = "4px"
  coordinates.style.color = "#475569"
  coordinates.style.fontSize = "12px"
  coordinates.textContent = `${markerData.lat.toFixed(6)}, ${markerData.lng.toFixed(6)}`
  wrapper.appendChild(coordinates)

  return wrapper
}

function getWorstHourlyLos(location: PTSILocation) {
  const hourlyLosValues = location.hourlyScores
    .map((hourlyScore) => resolveLosGrade(hourlyScore.los, hourlyScore.score, hourlyScore.losDescription))
    .filter((los): los is string => los !== null)

  if (hourlyLosValues.length === 0) {
    return null
  }

  return hourlyLosValues.reduce((worst, current) =>
    LOS_RANKS[current] > LOS_RANKS[worst] ? current : worst,
  )
}

function getHourlyLosForFocusTime(location: PTSILocation, focusTime: string) {
  const focusHour = extractHourOfDay(focusTime)
  if (focusHour === null) {
    return null
  }

  const hourlyEntries = location.hourlyScores
    .map((hourlyScore) => ({
      hour: extractHourOfDay(hourlyScore.hour),
      los: resolveLosGrade(hourlyScore.los, hourlyScore.score, hourlyScore.losDescription),
    }))
    .filter((entry): entry is { hour: number; los: string } => entry.hour !== null && entry.los !== null)

  const exactMatch = hourlyEntries.find((entry) => entry.hour === focusHour)
  if (exactMatch) {
    return { los: exactMatch.los, method: "focusTime exact" as const }
  }
  return null
}

function isMinuteWithinWindow(hourMinute: number, windowStartMinute: number, durationMinutes: number) {
  if (durationMinutes >= 1440) {
    return true
  }

  const normalizedHourMinute = normalizeMinutesOfDay(hourMinute)
  const normalizedWindowStart = normalizeMinutesOfDay(windowStartMinute)
  const normalizedWindowEnd = normalizeMinutesOfDay(normalizedWindowStart + durationMinutes)

  if (normalizedWindowStart < normalizedWindowEnd) {
    return normalizedHourMinute >= normalizedWindowStart && normalizedHourMinute < normalizedWindowEnd
  }

  return normalizedHourMinute >= normalizedWindowStart || normalizedHourMinute < normalizedWindowEnd
}

function getAverageHourlyLosForWindow(location: PTSILocation, timeRange?: string, startTime?: string) {
  const durationMinutes = durationFromTimeRange(timeRange)
  const windowStartMinutes = parseClockMinutes(startTime)
  if (durationMinutes === null || windowStartMinutes === null) {
    return null
  }

  const hourlyRanks = location.hourlyScores
    .map((hourlyScore) => {
      const minuteOfDay = parseClockMinutes(hourlyScore.hour)
      const los = resolveLosGrade(hourlyScore.los, hourlyScore.score, hourlyScore.losDescription)
      if (minuteOfDay === null || los === null) {
        return null
      }

      return { minuteOfDay, rank: LOS_RANKS[los] }
    })
    .filter((entry): entry is { minuteOfDay: number; rank: number } => entry !== null)
    .filter((entry) => isMinuteWithinWindow(entry.minuteOfDay, windowStartMinutes, durationMinutes))

  if (hourlyRanks.length === 0) {
    return null
  }

  const averageRank = hourlyRanks.reduce((sum, entry) => sum + entry.rank, 0) / hourlyRanks.length
  const roundedRank = Math.round(averageRank)
  return LOS_GRADES[roundedRank] ?? null
}

type LosResolution = {
  los: string | null
  losSource: "focusTime exact" | "window average" | "location LOS/score" | "worst hourly" | "unresolved"
  unresolvedReason?: string
}

function resolveLosFromLocation(location: PTSILocation, options: { focusTime?: string; timeRange?: string; startTime?: string }) {
  const { focusTime, timeRange, startTime } = options
  const hasWindowContext = Boolean(
    focusTime
    || (durationFromTimeRange(timeRange) !== null && parseClockMinutes(startTime) !== null),
  )

  if (focusTime) {
    const focusedHourlyLos = getHourlyLosForFocusTime(location, focusTime)
    if (focusedHourlyLos) {
      return {
        los: focusedHourlyLos.los,
        losSource: focusedHourlyLos.method,
      } satisfies LosResolution
    }
  }

  const averagedWindowLos = getAverageHourlyLosForWindow(location, timeRange, startTime)
  if (averagedWindowLos) {
    return {
      los: averagedWindowLos,
      losSource: "window average",
    } satisfies LosResolution
  }

  if (hasWindowContext) {
    return {
      los: null,
      losSource: "unresolved",
      unresolvedReason: "no LOS available inside selected time context",
    } satisfies LosResolution
  }

  const directLos = resolveLosGrade(location.los, location.score, location.losDescription)
  if (directLos) {
    return {
      los: directLos,
      losSource: "location LOS/score",
    } satisfies LosResolution
  }

  const worstHourlyLos = getWorstHourlyLos(location)
  if (worstHourlyLos) {
    return {
      los: worstHourlyLos,
      losSource: "worst hourly",
    } satisfies LosResolution
  }

  const hasHourlyEntries = location.hourlyScores.length > 0
  const hasAnyHourlyLos = location.hourlyScores.some(
    (hourlyScore) => resolveLosGrade(hourlyScore.los, hourlyScore.score, hourlyScore.losDescription) !== null,
  )
  const unresolvedReason = hasHourlyEntries
    ? hasAnyHourlyLos
      ? "hourly LOS exists but could not be resolved for selected context"
      : "hourly entries missing LOS/score-derived values and location LOS/score is missing"
    : "no hourly entries and no location LOS/score"

  return {
    los: null,
    losSource: "unresolved",
    unresolvedReason,
  } satisfies LosResolution
}

function squaredDistance(lat1: number, lng1: number, lat2: number, lng2: number) {
  const dLat = lat1 - lat2
  const dLng = lng1 - lng2
  return (dLat * dLat) + (dLng * dLng)
}

function colorForLos(los: string | null) {
  if (los === "A") return "#22C55E"
  if (los === "B") return "#84CC16"
  if (los === "C") return "#EAB308"
  if (los === "D") return "#F97316"
  if (los === "E") return "#EF4444"
  if (los === "F") return "#B91C1C"

  return "#94A3B8"
}

function getHeatmapStyleFromLos(losLabel: string) {
  const grade = normalizeLosGrade(losLabel.replace(/^LOS\s+/i, ""))
  const rank = grade ? LOS_RANKS[grade] : null
  const severity = rank === null ? 0.35 : (rank + 1) / LOS_GRADES.length

  return {
    outerRadiusMeters: 50 + (severity * 95),
    innerRadiusMeters: 22 + (severity * 48),
    outerOpacity: 0.08 + (severity * 0.24),
    innerOpacity: 0.18 + (severity * 0.42),
  }
}

function normalizeLocationName(name: string) {
  return name.toLowerCase().replace(/[^a-z0-9.]+/g, "").trim()
}

type MatchedLocation = {
  location: PTSILocation
  matchMethod: "selected" | "id" | "name" | "nearest"
}

function matchLocationForLandmark(options: {
  landmark: (typeof LANDMARKS)[number]
  locationById: Map<string, PTSILocation>
  locationByName: Map<string, PTSILocation>
  locationsWithValidCoords: PTSILocation[]
  selectedLocationRecord?: PTSILocation | null
  selectedLandmarkId?: string | null
}) {
  const {
    landmark,
    locationById,
    locationByName,
    locationsWithValidCoords,
    selectedLocationRecord,
    selectedLandmarkId,
  } = options

  if (selectedLocationRecord && selectedLandmarkId === landmark.id) {
    return {
      location: selectedLocationRecord,
      matchMethod: "selected",
    } satisfies MatchedLocation
  }

  const idMatch = locationById.get(landmark.id)
  if (idMatch) {
    return {
      location: idMatch,
      matchMethod: "id",
    } satisfies MatchedLocation
  }

  const nameMatch = locationByName.get(normalizeLocationName(landmark.name))
  if (nameMatch) {
    return {
      location: nameMatch,
      matchMethod: "name",
    } satisfies MatchedLocation
  }

  const nearestLocation = locationsWithValidCoords.reduce<PTSILocation | null>((closest, location) => {
    if (!closest) {
      return location
    }

    const currentDistance = squaredDistance(landmark.lat, landmark.lng, location.latitude, location.longitude)
    const closestDistance = squaredDistance(landmark.lat, landmark.lng, closest.latitude, closest.longitude)
    return currentDistance < closestDistance ? location : closest
  }, null)

  if (!nearestLocation) {
    return null
  }

  return {
    location: nearestLocation,
    matchMethod: "nearest",
  } satisfies MatchedLocation
}

function resolveSelectedLandmarkId(
  selectedLocationId: string | undefined,
  selectedLocationRecord: PTSILocation | null,
  locationByName: Map<string, PTSILocation>,
  locationsWithValidCoords: PTSILocation[],
) {
  if (!selectedLocationId) {
    return null
  }

  const byLandmarkId = LANDMARKS.find((landmark) => landmark.id === selectedLocationId)
  if (byLandmarkId) {
    return byLandmarkId.id
  }

  if (!selectedLocationRecord) {
    return null
  }

  const byLocationId = LANDMARKS.find((landmark) => landmark.id === selectedLocationRecord.id)
  if (byLocationId) {
    return byLocationId.id
  }

  const selectedLocationNameKey = normalizeLocationName(selectedLocationRecord.name)
  const nameMatchedLocation = locationByName.get(selectedLocationNameKey)
  if (nameMatchedLocation) {
    const byName = LANDMARKS.find((landmark) => normalizeLocationName(landmark.name) === selectedLocationNameKey)
    if (byName) {
      return byName.id
    }
  }

  if (!Number.isFinite(selectedLocationRecord.latitude) || !Number.isFinite(selectedLocationRecord.longitude)) {
    return null
  }

  const nearestLandmark = LANDMARKS.reduce<(typeof LANDMARKS)[number] | null>((closest, landmark) => {
    if (!closest) {
      return landmark
    }

    const currentDistance = squaredDistance(landmark.lat, landmark.lng, selectedLocationRecord.latitude, selectedLocationRecord.longitude)
    const closestDistance = squaredDistance(closest.lat, closest.lng, selectedLocationRecord.latitude, selectedLocationRecord.longitude)
    return currentDistance < closestDistance ? landmark : closest
  }, null)

  return nearestLandmark?.id ?? null
}

export function CampusOsmMap({
  selectedDate,
  occlusionData,
  focusTime,
  timeRange,
  startTime,
  zoomLevel,
  selectedLocationId,
  showLosDetails = true,
  className,
}: CampusOsmMapProps) {
  const mapHostRef = useRef<HTMLDivElement | null>(null)
  const dateLabel = useMemo(() => formatDateLabel(selectedDate), [selectedDate])
  const mapContextLabel = useMemo(() => {
    if (!focusTime) {
      const readableRange = formatTimeRangeLabel(timeRange)
      if (startTime) {
        return `${readableRange} from ${startTime}`
      }

      return readableRange
    }

    const zoomContext = typeof zoomLevel === "number" && zoomLevel > 0 ? ` (zoom ${zoomLevel})` : ""
    return `${focusTime}${zoomContext}`
  }, [focusTime, startTime, timeRange, zoomLevel])

  const markers = useMemo(() => {
    const locationCandidates = occlusionData?.locations ?? []
    const locationById = new Map(locationCandidates.map((location) => [location.id, location]))
    const locationByName = new Map(locationCandidates.map((location) => [normalizeLocationName(location.name), location]))
    const locationsWithValidCoords = locationCandidates.filter(
      (location) => Number.isFinite(location.latitude) && Number.isFinite(location.longitude),
    )
    const selectedLocationRecord = selectedLocationId ? locationById.get(selectedLocationId) ?? null : null
    const selectedLandmarkId = resolveSelectedLandmarkId(
      selectedLocationId,
      selectedLocationRecord,
      locationByName,
      locationsWithValidCoords,
    )

    return LANDMARKS.map((landmark) => {
      const matchedLocation = matchLocationForLandmark({
        landmark,
        locationById,
        locationByName,
        locationsWithValidCoords,
        selectedLocationRecord,
        selectedLandmarkId,
      })
      const losResolution = matchedLocation
        ? resolveLosFromLocation(matchedLocation.location, {
            focusTime,
            timeRange,
            startTime,
          })
        : null
      const isSelected = Boolean(
        selectedLocationId
        && (matchedLocation?.location.id === selectedLocationId || landmark.id === selectedLocationId),
      )

      const resolvedLos = losResolution?.los ?? null
      const markerLos = resolvedLos ? `LOS ${resolvedLos}` : "LOS -"
      const matchLabel = matchedLocation
        ? matchedLocation.matchMethod === "selected"
          ? `selected location (${matchedLocation.location.name})`
          : matchedLocation.matchMethod === "id"
            ? `ID match (${matchedLocation.location.name})`
            : matchedLocation.matchMethod === "name"
              ? `name match (${matchedLocation.location.name})`
              : `nearest-coordinate match (${matchedLocation.location.name})`
        : "no location match"
      const losLabel = losResolution
        ? losResolution.los
          ? losResolution.losSource
          : `unresolved: ${losResolution.unresolvedReason ?? "missing hourly/location LOS"}`
        : "unresolved: no matched occlusion location"
      const sourceLabel = `Match: ${matchLabel} • LOS source: ${losLabel}`

      return {
        id: landmark.id,
        name: landmark.name,
        lat: landmark.lat,
        lng: landmark.lng,
        los: markerLos,
        markerColor: showLosDetails ? colorForLos(resolvedLos) : "#0EA5E9",
        isSelected,
        sourceLabel,
      }
    })
  }, [focusTime, occlusionData, selectedLocationId, showLosDetails, startTime, timeRange])

  useEffect(() => {
    if (!mapHostRef.current) {
      return
    }

    let map: LeafletMap | null = null
    let isCancelled = false
    let resizeObserver: ResizeObserver | null = null
    let delayedInvalidateTimer: number | null = null

    const mapBounds: LatLngBoundsExpression = [
      [CAMPUS_MAP_BOUNDS.south, CAMPUS_MAP_BOUNDS.west],
      [CAMPUS_MAP_BOUNDS.north, CAMPUS_MAP_BOUNDS.east],
    ]

    void (async () => {
      const L = await import("leaflet")
      if (isCancelled || !mapHostRef.current) {
        return
      }

      map = L.map(mapHostRef.current, {
        minZoom: 16,
        maxZoom: 20,
        maxBounds: mapBounds,
        maxBoundsViscosity: 1,
        zoomControl: true,
        scrollWheelZoom: true,
      })

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 20,
        minZoom: 16,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(map)

      if (markers.length > 0) {
        const markerBounds: LatLngBoundsExpression = markers.map((marker) => [marker.lat, marker.lng])
        map.fitBounds(markerBounds, { padding: [24, 24], maxZoom: 18 })
      } else {
        map.fitBounds(mapBounds, { padding: [12, 12] })
      }

      const safelyInvalidateSize = () => {
        if (!map || isCancelled) {
          return
        }

        map.invalidateSize()
      }

      requestAnimationFrame(safelyInvalidateSize)
      delayedInvalidateTimer = window.setTimeout(safelyInvalidateSize, 180)

      resizeObserver = new ResizeObserver(() => {
        safelyInvalidateSize()
      })
      resizeObserver.observe(mapHostRef.current)

      if (!map) {
        return
      }

      const activeMap = map

      markers.forEach((markerData) => {
        const heatmapStyle = getHeatmapStyleFromLos(markerData.los)

        L.circle([markerData.lat, markerData.lng], {
          radius: heatmapStyle.outerRadiusMeters,
          stroke: false,
          fillColor: markerData.markerColor,
          fillOpacity: heatmapStyle.outerOpacity,
          interactive: false,
        }).addTo(activeMap)

        L.circle([markerData.lat, markerData.lng], {
          radius: heatmapStyle.innerRadiusMeters,
          stroke: false,
          fillColor: markerData.markerColor,
          fillOpacity: heatmapStyle.innerOpacity,
          interactive: false,
        }).addTo(activeMap)

        const marker = L.circleMarker([markerData.lat, markerData.lng], {
          radius: markerData.isSelected ? 11 : 8,
          color: markerData.isSelected ? "#0F172A" : "#E2E8F0",
          weight: markerData.isSelected ? 3 : 1.5,
          fillColor: markerData.markerColor,
          fillOpacity: markerData.isSelected ? 1 : 0.95,
        }).addTo(activeMap)

        const tooltipContent = `<strong>${showLosDetails ? markerData.los : markerData.name}</strong>`
        marker.bindTooltip(tooltipContent, {
          direction: "top",
          offset: [0, -8],
          permanent: false,
          opacity: 0.95,
          className: "campus-osm-marker-label",
        })

        const popupContent = showLosDetails
          ? buildMarkerPopupContent(markerData, dateLabel, mapContextLabel)
          : (() => {
              const wrapper = document.createElement("div")
              wrapper.style.fontFamily = "Inter, system-ui, sans-serif"
              const title = document.createElement("div")
              title.style.fontWeight = "700"
              title.textContent = markerData.name
              wrapper.appendChild(title)
              return wrapper
            })()
        marker.bindPopup(popupContent)
        marker.on("mouseover", () => marker.openTooltip())
        marker.on("click", () => {
          const targetZoom = Math.max(activeMap.getZoom(), 18)
          activeMap.flyTo([markerData.lat, markerData.lng], targetZoom, { duration: 0.45 })
          marker.openPopup()
        })
      })
    })()

    return () => {
      isCancelled = true
      if (delayedInvalidateTimer !== null) {
        window.clearTimeout(delayedInvalidateTimer)
      }
      resizeObserver?.disconnect()
      map?.remove()
    }
  }, [dateLabel, mapContextLabel, markers, showLosDetails])

  return (
    <div className={`${className ?? "h-72 w-full rounded-xl"} relative overflow-hidden`}>
      <div ref={mapHostRef} className="h-full w-full" />
      {showLosDetails && (
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
      )}
      <style>{`
        .campus-osm-marker-label {
          background: rgba(15, 23, 42, 0.88);
          border: 1px solid rgba(148, 163, 184, 0.45);
          border-radius: 9999px;
          color: #f8fafc;
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.02em;
          padding: 2px 8px;
          box-shadow: 0 4px 14px rgba(15, 23, 42, 0.25);
        }

        .campus-osm-marker-label::before {
          border-top-color: rgba(15, 23, 42, 0.88) !important;
        }
      `}</style>
    </div>
  )
}
