"use client"

import { MapPin } from "lucide-react"
import { CampusOsmMap } from "@/components/maps/campus-osm-map"

interface Location {
  id: string
  name: string
  address?: string
  latitude: number
  longitude: number
  videos: Array<{ id: string }>
}

interface LocationMapProps {
  locations: Location[]
  selectedDate?: string
}

export function LocationMap({ locations, selectedDate }: LocationMapProps) {
  return (
    <div className="rounded-2xl border border-border bg-secondary/50 p-3 shadow-elevated-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Campus Landmark Map</h3>
        <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
          {locations.length} {locations.length === 1 ? "location" : "locations"}
        </span>
      </div>

      <CampusOsmMap selectedDate={selectedDate} showLosDetails={false} className="h-[clamp(14rem,36vh,24rem)] w-full rounded-xl border border-border" />

      {locations.length > 0 && (
        <div className="mt-2.5 border-t border-border/60 pt-2.5">
          <div className="flex flex-wrap gap-1.5">
            {locations.map((location) => (
              <div
                key={location.id}
                className="inline-flex items-center gap-1.5 rounded-lg border border-border/70 bg-background/50 px-2.5 py-1 text-[11px]"
              >
                <MapPin className="h-2.5 w-2.5 shrink-0 text-muted-foreground" />
                <span className="font-medium text-foreground">{location.name}</span>
                <span className="text-muted-foreground">
                  · {location.videos.length} {location.videos.length === 1 ? "feed" : "feeds"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
