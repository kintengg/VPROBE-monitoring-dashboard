"use client"

import { useMemo } from "react"
import { useRouter } from "next/navigation"
import { Car, Loader2 } from "lucide-react"
import type { EventRecord } from "@/lib/api"

interface VehicleEventFeedProps {
  events?: EventRecord[]
  loading?: boolean
  emptyMessage?: string
}

export function VehicleEventFeed({
  events = [],
  loading = false,
  emptyMessage = "No vehicle detections yet. Run a video through the RT-DETR pipeline to populate this feed.",
}: VehicleEventFeedProps) {
  const router = useRouter()

  const displayEvents = useMemo(() => [...events].reverse(), [events])

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="text-sm font-medium text-foreground">Vehicle Event Feed</h3>
        <p className="text-xs text-muted-foreground mt-0.5">
          Recent gate detections from RT-DETR
        </p>
      </div>

      <div className="flex-1 overflow-auto">
        {loading ? (
          <div className="flex items-center justify-center py-10 text-muted-foreground">
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            <span className="text-sm">Loading events…</span>
          </div>
        ) : displayEvents.length === 0 ? (
          <div className="flex flex-col items-center justify-center px-6 py-12 text-center text-muted-foreground">
            <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-2xl bg-secondary">
              <Car className="h-5 w-5" />
            </div>
            <p className="text-xs leading-relaxed">{emptyMessage}</p>
          </div>
        ) : (
          <ul className="divide-y divide-border">
            {displayEvents.map((event) => (
              <li
                key={event.id}
                className="cursor-pointer px-4 py-3 transition-colors hover:bg-secondary/40"
                onClick={() => {
                  if (event.videoId) {
                    router.push(`/video/${event.videoId}`)
                  }
                }}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-foreground truncate">
                      {event.gateName ?? event.location ?? "Unknown gate"}
                    </p>
                    <p className="mt-0.5 text-xs text-muted-foreground line-clamp-2">
                      {event.description ?? event.type}
                    </p>
                  </div>
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground shrink-0">
                    {event.vehicleClass ?? event.type}
                  </span>
                </div>
                {typeof event.offsetSeconds === "number" && (
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    +{event.offsetSeconds.toFixed(1)}s
                  </p>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
