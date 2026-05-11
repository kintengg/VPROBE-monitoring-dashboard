"use client"

import { useMemo, useState } from "react"
import { useRouter } from "next/navigation"
import { AlertCircle, ArrowRight, Bike, BusFront, CarFront, Loader2, Truck, Van } from "lucide-react"
import type { LucideIcon } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import type { EventRecord } from "@/lib/api"

const EVENT_FEED_LIMIT = 50

interface EventFeedProps {
  filteredVideoId?: string
  events?: EventRecord[]
  loading?: boolean
  selectedEventId?: string
  onEventSelect?: (event: EventRecord) => void
}

export function EventFeed({ filteredVideoId, events = [], loading = false, selectedEventId, onEventSelect }: EventFeedProps) {
  const router = useRouter()
  const [selectedVehicleType, setSelectedVehicleType] = useState<string>("all")

  const vehicleTypeOptions = useMemo(() => {
    const types = new Set<string>()
    for (const event of events) {
      if (event.type !== "detection") {
        continue
      }
      types.add(resolveVehicleType(event))
    }

    return ["all", ...Array.from(types).sort((left, right) => left.localeCompare(right))]
  }, [events])

  const displayEvents = useMemo(() => {
    const filteredEvents = selectedVehicleType === "all"
      ? events
      : events.filter((event) => event.type === "detection" && resolveVehicleType(event) === selectedVehicleType)

    const sorted = filteredVideoId
      ? [...filteredEvents].sort((left, right) => {
          const leftOffset = typeof left.offsetSeconds === "number" ? left.offsetSeconds : Number.POSITIVE_INFINITY
          const rightOffset = typeof right.offsetSeconds === "number" ? right.offsetSeconds : Number.POSITIVE_INFINITY
          return leftOffset - rightOffset
        })
      : [...filteredEvents].reverse()

    // Limit to EVENT_FEED_LIMIT when not inside a specific video
    return filteredVideoId ? sorted : sorted.slice(0, EVENT_FEED_LIMIT)
  }, [events, filteredVideoId, selectedVehicleType])


  const handleEventSelect = (event: EventRecord) => {
    if (onEventSelect) {
      onEventSelect(event)
      return
    }

    if (!event.videoId) {
      return
    }

    const params = new URLSearchParams({ eventId: event.id })
    if (typeof event.offsetSeconds === "number") {
      params.set("seek", String(event.offsetSeconds))
    }

    const query = params.toString()
    router.push(query ? `/video/${event.videoId}?${query}` : `/video/${event.videoId}`)
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="text-sm font-medium text-foreground">Event Feed</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          {filteredVideoId
            ? "Click an event to seek within this recording."
            : `Showing the ${EVENT_FEED_LIMIT} most recent detections. Click to open the relevant footage.`}
        </p>
        <div className="mt-3">
          <Select value={selectedVehicleType} onValueChange={setSelectedVehicleType}>
            <SelectTrigger className="h-8 bg-secondary border-border text-xs text-foreground">
              <SelectValue placeholder="Filter by vehicle type" />
            </SelectTrigger>
            <SelectContent className="bg-card border-border">
              {vehicleTypeOptions.map((type) => (
                <SelectItem key={type} value={type} className="text-xs text-foreground">
                  {type === "all" ? "All vehicle types" : type}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        {loading ? (
          <div className="flex items-center justify-center h-full p-6 text-muted-foreground">
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            Loading events...
          </div>
        ) : displayEvents.length > 0 ? (
          <div className="p-2 space-y-2">
            {displayEvents.map((event, index) => (
              <EventCard
                key={`${event.id}-${index}`}
                event={event}
                active={selectedEventId === event.id}
                interactive={Boolean(onEventSelect || event.videoId)}
                onSelect={() => handleEventSelect(event)}
              />
            ))}
          </div>
        ) : (
          <div className="p-6 text-sm text-muted-foreground">
            No events available yet.
          </div>
        )}
      </div>
    </div>
  )
}

function formatOffset(event: EventRecord) {
  if (typeof event.offsetSeconds === "number" && Number.isFinite(event.offsetSeconds)) {
    const minutes = Math.floor(event.offsetSeconds / 60)
    const seconds = Math.floor(event.offsetSeconds % 60)
    return `${minutes}:${seconds.toString().padStart(2, "0")}`
  }

  if (typeof event.frame === "number") {
    return `Frame ${event.frame}`
  }

  return "--"
}

function formatVehicleClassLabel(value: string) {
  return value
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(" ")
}

function resolveVehicleType(event: EventRecord) {
  const backendLabel = (event.vehicleClassLabel ?? "").trim()
  if (backendLabel) {
    return backendLabel
  }

  const backendClass = (event.vehicleClass ?? "").trim()
  if (backendClass) {
    return formatVehicleClassLabel(backendClass)
  }

  return "Unclassified"
}

function resolveVehicleIcon(vehicleType: string): LucideIcon {
  const normalized = vehicleType.trim().toLowerCase()

  if (normalized.includes("bus")) return BusFront
  if (normalized.includes("truck")) return Truck
  if (normalized.includes("van") || normalized.includes("suv")) return Van
  if (normalized.includes("motor") || normalized.includes("tricycle")) return Bike
  if (normalized.includes("bicycle") || normalized.includes("bike")) return Bike
  if (normalized.includes("jeep")) return CarFront
  if (normalized.includes("car") || normalized.includes("taxi")) return CarFront

  return CarFront
}

function formatEventDescription(description: string, vehicleType: string) {
  return description
    .replace(/Pedestrian ID/gi, "Vehicle ID")
    .replace(/\bPedestrian\b/gi, "Vehicle")
    .replace(/Vehicle ID\s*#?\d+/gi, vehicleType)
}

function EventCard({
  event,
  active,
  interactive,
  onSelect,
}: {
  event: EventRecord
  active: boolean
  interactive: boolean
  onSelect: () => void
}) {
  const isDetection = event.type === "detection"
  const vehicleType = resolveVehicleType(event)
  const VehicleIcon = resolveVehicleIcon(vehicleType)

  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={!interactive}
      className={[
        "w-full rounded-2xl border p-3 text-left transition-all",
        active ? "border-primary bg-primary/10 shadow-elevated-sm" : "border-border bg-secondary/50 hover:border-primary/30 hover:bg-secondary",
        interactive ? "cursor-pointer" : "cursor-default opacity-80",
      ].join(" ")}
    >
      <div className="flex items-start gap-3">
        <div className="w-14 h-10 rounded-xl bg-[#1C1C1E] flex items-center justify-center shrink-0">
          {isDetection ? <VehicleIcon className="w-4 h-4 text-accent" /> : <AlertCircle className="w-4 h-4 text-chart-4" />}
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-2">
            <p className="text-sm font-medium text-foreground truncate">{event.location}</p>
            {interactive && <ArrowRight className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />}
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground">{formatEventDescription(event.description, vehicleType)}</p>
          <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[10px]">
            <span className="rounded-full bg-muted px-2 py-0.5 text-muted-foreground">{event.timestamp}</span>
            <span className="rounded-full bg-primary/10 px-2 py-0.5 text-primary">{vehicleType}</span>
            {typeof event.pedestrianId === "number" && (
              <span className="rounded-full bg-accent/10 px-2 py-0.5 font-medium text-accent">Vehicle ID #{event.pedestrianId}</span>
            )}
          </div>
        </div>
      </div>
    </button>
  )
}
