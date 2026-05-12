"use client"

import { useEffect, useRef, useState } from "react"
import { MapPin } from "lucide-react"
import type { Map as LeafletMap, LayerGroup as LeafletLayerGroup, Layer as LeafletLayer } from "leaflet"
import { LOS_UNKNOWN_HEX } from "@/lib/los-colors"

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
  /** Marker glyph — defaults to "triangle" (pedestrian); "circle" for vehicle gates. */
  markerShape?: "triangle" | "circle"
}

// Same triangle div-icon style used by the consolidated/unified LOS map so that
// pedestrian-side maps (dashboard + overview) read identically.
function triangleDivIconHtml(color: string, size: number): string {
  return `
    <div style="width:${size}px; height:${size}px; filter:drop-shadow(0 1px 1px rgba(0,0,0,0.4));">
      <svg viewBox="0 0 24 24" width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
        <polygon points="12,2 22,21 2,21" fill="${color}" stroke="#ffffff" stroke-width="2" stroke-linejoin="round" />
      </svg>
    </div>
  `
}

export function LocationMap({ locations, selectedDate, markerShape = "triangle" }: LocationMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<LeafletMap | null>(null)
  const layerRef = useRef<LeafletLayerGroup | null>(null)
  const markersRef = useRef<Map<string, LeafletLayer>>(new Map())
  const [leafletReady, setLeafletReady] = useState(false)
  const leafletModuleRef = useRef<typeof import("leaflet") | null>(null)

  // Initialise the Leaflet map once on mount.
  useEffect(() => {
    if (!containerRef.current || typeof window === "undefined") return
    let cancelled = false

    void (async () => {
      const L = await import("leaflet")
      if (cancelled || !containerRef.current) return
      leafletModuleRef.current = L
      const map = L.map(containerRef.current, {
        zoomControl: true,
        scrollWheelZoom: true,
        attributionControl: true,
      })
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 20,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(map)
      layerRef.current = L.layerGroup().addTo(map)
      mapRef.current = map
      // Default view fits Ateneo campus until we have markers to fit on.
      map.setView([14.6390, 121.0770], 17)
      setLeafletReady(true)
    })()

    return () => {
      cancelled = true
      markersRef.current.clear()
      if (layerRef.current) layerRef.current.clearLayers()
      layerRef.current = null
      mapRef.current?.remove()
      mapRef.current = null
    }
  }, [])

  // Sync the triangle markers whenever the locations list changes.
  useEffect(() => {
    if (!leafletReady) return
    const L = leafletModuleRef.current
    const map = mapRef.current
    const layer = layerRef.current
    if (!L || !map || !layer) return

    const validLocations = locations.filter(
      (loc) => Number.isFinite(loc.latitude) && Number.isFinite(loc.longitude),
    )
    const present = new Set(validLocations.map((loc) => loc.id))

    // Drop markers that no longer correspond to a location.
    for (const [id, marker] of markersRef.current.entries()) {
      if (!present.has(id)) {
        layer.removeLayer(marker)
        markersRef.current.delete(id)
      }
    }

    const size = 22
    for (const location of validLocations) {
      const popupHtml = `
        <div style="font-family: Inter, system-ui, sans-serif; min-width: 160px;">
          <div style="font-weight: 700; font-size: 13px; margin-bottom: 4px;">${location.name}</div>
          ${location.address ? `<div style="font-size: 11px; color: #64748b; margin-bottom: 4px;">${location.address}</div>` : ""}
          <div style="font-size: 11px; color: #64748b;">${location.latitude.toFixed(6)}, ${location.longitude.toFixed(6)}</div>
          <div style="font-size: 11px; color: #64748b; margin-top: 4px;">${location.videos.length} ${location.videos.length === 1 ? "feed" : "feeds"}${selectedDate ? ` on ${selectedDate}` : ""}</div>
        </div>
      `

      // Always rebuild the marker so a shape change between renders takes effect cleanly.
      const existing = markersRef.current.get(location.id)
      if (existing) {
        layer.removeLayer(existing)
        markersRef.current.delete(location.id)
      }

      const marker = markerShape === "circle"
        ? L.circleMarker([location.latitude, location.longitude], {
            radius: 9,
            color: "#ffffff",
            weight: 2,
            fillColor: LOS_UNKNOWN_HEX,
            fillOpacity: 0.95,
          })
        : L.marker([location.latitude, location.longitude], {
            icon: L.divIcon({
              className: "pedestrian-location-triangle",
              html: triangleDivIconHtml(LOS_UNKNOWN_HEX, size),
              iconSize: [size, size],
              iconAnchor: [size / 2, size - 2],
              popupAnchor: [0, -size + 2],
            }),
          })
      marker.bindPopup(popupHtml).bindTooltip(location.name, { direction: "top", offset: [0, -size] })
      layer.addLayer(marker)
      markersRef.current.set(location.id, marker)
    }

    // Fit bounds to the markers when we have at least one.
    if (validLocations.length > 0) {
      const bounds = L.latLngBounds(validLocations.map((loc) => [loc.latitude, loc.longitude]))
      map.fitBounds(bounds, { padding: [32, 32], maxZoom: 18 })
    }
  }, [leafletReady, locations, markerShape, selectedDate])

  return (
    <div className="rounded-2xl border border-border bg-secondary/50 p-3 shadow-elevated-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">Campus Landmark Map</h3>
        <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
          {locations.length} {locations.length === 1 ? "location" : "locations"}
        </span>
      </div>

      <div
        ref={containerRef}
        className="h-[clamp(14rem,36vh,24rem)] w-full rounded-xl border border-border"
      />

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
