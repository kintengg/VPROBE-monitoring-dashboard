"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import type {
  CircleMarker as LeafletCircleMarker,
  LayerGroup as LeafletLayerGroup,
  Map as LeafletMap,
  Marker as LeafletMarker,
} from "leaflet"
import type { PTSILocation, VehicleGate, VehicleGateLOS, VehicleLOS } from "@/lib/api"
import { LOS_GRADES, LOS_HEX, LOS_UNKNOWN_HEX } from "@/lib/los-colors"

const CAMPUS_BOUNDS = [
  [14.63155, 121.07088],
  [14.64807, 121.08418],
] as const

const TILE_LAYER_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
const TILE_LAYER_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'

interface UnifiedLosMapProps {
  vehicleGates: VehicleGate[]
  vehicleLos?: Record<string, VehicleGateLOS>
  pedestrianLocations: PTSILocation[]
  height?: string
  className?: string
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
}

function vehiclePopupHtml(gate: VehicleGate, los: VehicleGateLOS | undefined): string {
  const flow = gate.flowGroup === "In" ? "Entrance" : "Exit"
  const hasData = los != null && los.vehicleCount > 0
  const losDisplay = hasData ? (los.los ?? "—") : "—"
  const vc = hasData && los?.vcRatio != null ? los.vcRatio.toFixed(2) : "—"
  return `
    <div style="font-family: inherit; min-width: 180px;">
      <div style="font-weight:600; font-size:13px; margin-bottom:4px;">🚗 ${escapeHtml(gate.name)}</div>
      <div style="font-size:11px; color:#64748b; margin-bottom:6px;">${flow} gate</div>
      <div style="display:grid; grid-template-columns: auto auto; gap:4px 12px; font-size:12px;">
        <span style="color:#64748b">LOS</span><span><strong>${losDisplay}</strong></span>
        <span style="color:#64748b">V/C</span><span>${vc}</span>
        <span style="color:#64748b">Vehicles</span><span>${los?.vehicleCount ?? 0}</span>
      </div>
    </div>
  `
}

function pedestrianPopupHtml(location: PTSILocation): string {
  const hasData = location.averagePedestrians != null && location.averagePedestrians > 0
  const losDisplay = hasData ? (location.los ?? "—") : "—"
  const score = hasData && location.score != null ? location.score.toFixed(1) : "—"
  return `
    <div style="font-family: inherit; min-width: 180px;">
      <div style="font-weight:600; font-size:13px; margin-bottom:4px;">🚶 ${escapeHtml(location.name)}</div>
      <div style="font-size:11px; color:#64748b; margin-bottom:6px;">Pedestrian walkway</div>
      <div style="display:grid; grid-template-columns: auto auto; gap:4px 12px; font-size:12px;">
        <span style="color:#64748b">LOS</span><span><strong>${losDisplay}</strong></span>
        <span style="color:#64748b">Score</span><span>${score}</span>
        <span style="color:#64748b">Avg ped</span><span>${location.averagePedestrians ?? 0}</span>
      </div>
    </div>
  `
}

function triangleDivIconHtml(color: string, size: number, isSelected: boolean): string {
  const shadow = isSelected ? "drop-shadow(0 0 6px rgba(255,255,255,0.85))" : "drop-shadow(0 1px 1px rgba(0,0,0,0.4))"
  // Equilateral triangle pointing up. Stroke is white for contrast against tiles.
  return `
    <div style="width:${size}px; height:${size}px; filter:${shadow};">
      <svg viewBox="0 0 24 24" width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
        <polygon points="12,2 22,21 2,21" fill="${color}" stroke="#ffffff" stroke-width="2" stroke-linejoin="round" />
      </svg>
    </div>
  `
}

export function UnifiedLosMap({
  vehicleGates,
  vehicleLos = {},
  pedestrianLocations,
  height = "100%",
  className,
}: UnifiedLosMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<LeafletMap | null>(null)
  const vehicleLayerRef = useRef<LeafletLayerGroup | null>(null)
  const pedestrianLayerRef = useRef<LeafletLayerGroup | null>(null)
  const vehicleMarkers = useRef<Map<string, LeafletCircleMarker>>(new Map())
  const pedestrianMarkers = useRef<Map<string, LeafletMarker>>(new Map())
  const [leaflet, setLeaflet] = useState<typeof import("leaflet") | null>(null)
  const [showVehicles, setShowVehicles] = useState(true)
  const [showPedestrians, setShowPedestrians] = useState(true)

  // Load leaflet once
  useEffect(() => {
    let cancelled = false
    void import("leaflet").then((mod) => {
      if (!cancelled) setLeaflet((mod.default ?? mod) as typeof import("leaflet"))
    })
    return () => {
      cancelled = true
    }
  }, [])

  // Initialize map + layer groups
  useEffect(() => {
    if (!leaflet || !containerRef.current || mapRef.current) return
    const map = leaflet.map(containerRef.current, { minZoom: 14 }).fitBounds(CAMPUS_BOUNDS as unknown as [number, number][])
    leaflet
      .tileLayer(TILE_LAYER_URL, { attribution: TILE_LAYER_ATTRIBUTION, maxZoom: 19 })
      .addTo(map)
    vehicleLayerRef.current = leaflet.layerGroup().addTo(map)
    pedestrianLayerRef.current = leaflet.layerGroup().addTo(map)
    mapRef.current = map
    return () => {
      map.remove()
      mapRef.current = null
      vehicleLayerRef.current = null
      pedestrianLayerRef.current = null
      vehicleMarkers.current.clear()
      pedestrianMarkers.current.clear()
    }
  }, [leaflet])

  // Toggle vehicle layer visibility
  useEffect(() => {
    const map = mapRef.current
    const layer = vehicleLayerRef.current
    if (!map || !layer) return
    if (showVehicles && !map.hasLayer(layer)) layer.addTo(map)
    else if (!showVehicles && map.hasLayer(layer)) map.removeLayer(layer)
  }, [showVehicles])

  // Toggle pedestrian layer visibility
  useEffect(() => {
    const map = mapRef.current
    const layer = pedestrianLayerRef.current
    if (!map || !layer) return
    if (showPedestrians && !map.hasLayer(layer)) layer.addTo(map)
    else if (!showPedestrians && map.hasLayer(layer)) map.removeLayer(layer)
  }, [showPedestrians])

  // Sync vehicle markers (circles)
  useEffect(() => {
    const layer = vehicleLayerRef.current
    if (!leaflet || !layer) return

    const present = new Set(vehicleGates.map((g) => g.id))
    for (const [id, marker] of vehicleMarkers.current.entries()) {
      if (!present.has(id)) {
        layer.removeLayer(marker)
        vehicleMarkers.current.delete(id)
      }
    }

    for (const gate of vehicleGates) {
      const los = vehicleLos[gate.id]
      const hasData = los != null && los.vehicleCount > 0
      const fill = hasData && los.los ? LOS_HEX[los.los] : LOS_UNKNOWN_HEX
      const popup = vehiclePopupHtml(gate, los)

      const existing = vehicleMarkers.current.get(gate.id)
      if (existing) {
        existing.setLatLng([gate.latitude, gate.longitude])
        existing.setStyle({
          color: "#ffffff",
          fillColor: fill,
          fillOpacity: 0.9,
          weight: 2,
          radius: 11,
        })
        existing.setPopupContent(popup)
        continue
      }

      const marker = leaflet.circleMarker([gate.latitude, gate.longitude], {
        radius: 11,
        color: "#ffffff",
        weight: 2,
        fillColor: fill,
        fillOpacity: 0.9,
      })
      marker.bindPopup(popup).bindTooltip(gate.name, { direction: "top", offset: [0, -10] })
      layer.addLayer(marker)
      vehicleMarkers.current.set(gate.id, marker)
    }
  }, [leaflet, vehicleGates, vehicleLos])

  // Sync pedestrian markers (triangles via divIcon)
  useEffect(() => {
    const layer = pedestrianLayerRef.current
    if (!leaflet || !layer) return

    const present = new Set(pedestrianLocations.map((p) => p.id))
    for (const [id, marker] of pedestrianMarkers.current.entries()) {
      if (!present.has(id)) {
        layer.removeLayer(marker)
        pedestrianMarkers.current.delete(id)
      }
    }

    for (const location of pedestrianLocations) {
      const los = location.los as VehicleLOS | null | undefined
      const hasData = location.averagePedestrians != null && location.averagePedestrians > 0
      const color = hasData && los ? LOS_HEX[los] : LOS_UNKNOWN_HEX
      const size = 22
      const icon = leaflet.divIcon({
        className: "unified-los-triangle",
        html: triangleDivIconHtml(color, size, false),
        iconSize: [size, size],
        iconAnchor: [size / 2, size - 2],
        popupAnchor: [0, -size + 2],
      })
      const popup = pedestrianPopupHtml(location)

      const existing = pedestrianMarkers.current.get(location.id)
      if (existing) {
        existing.setLatLng([location.latitude, location.longitude])
        existing.setIcon(icon)
        existing.setPopupContent(popup)
        continue
      }

      const marker = leaflet.marker([location.latitude, location.longitude], { icon })
      marker.bindPopup(popup).bindTooltip(location.name, { direction: "top", offset: [0, -size] })
      layer.addLayer(marker)
      pedestrianMarkers.current.set(location.id, marker)
    }
  }, [leaflet, pedestrianLocations])

  const losCounts = useMemo(() => {
    const counts: Record<string, { ped: number; veh: number }> = {}
    for (const grade of LOS_GRADES) counts[grade] = { ped: 0, veh: 0 }
    for (const row of Object.values(vehicleLos)) {
      if (row.los && row.vehicleCount > 0) counts[row.los].veh += 1
    }
    for (const loc of pedestrianLocations) {
      if (loc.los && loc.averagePedestrians != null && loc.averagePedestrians > 0) counts[loc.los].ped += 1
    }
    return counts
  }, [vehicleLos, pedestrianLocations])

  return (
    <div className={className} style={{ position: "relative", height }}>
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
      {/* Legend / layer toggle */}
      <div
        className="absolute right-3 top-3 rounded-xl border border-border/60 bg-card/95 px-3 py-2.5 text-[11px] text-foreground shadow-elevated-sm"
        style={{ zIndex: 1000, minWidth: 200 }}
      >
        <div className="mb-2 text-[10px] uppercase tracking-wider text-muted-foreground">Layers</div>
        <div className="space-y-1.5">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showVehicles}
              onChange={(e) => setShowVehicles(e.target.checked)}
              className="h-3 w-3"
            />
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-slate-300" style={{ background: LOS_UNKNOWN_HEX, border: "1px solid #fff" }} />
            <span>Vehicle gates</span>
            <span className="ml-auto text-muted-foreground">{vehicleGates.length}</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={showPedestrians}
              onChange={(e) => setShowPedestrians(e.target.checked)}
              className="h-3 w-3"
            />
            <span
              className="inline-block"
              style={{
                width: 0,
                height: 0,
                borderLeft: "5px solid transparent",
                borderRight: "5px solid transparent",
                borderBottom: `9px solid ${LOS_UNKNOWN_HEX}`,
              }}
            />
            <span>Pedestrian walkways</span>
            <span className="ml-auto text-muted-foreground">{pedestrianLocations.length}</span>
          </label>
        </div>
        <div className="mt-3 mb-1 text-[10px] uppercase tracking-wider text-muted-foreground">LOS</div>
        <div className="grid grid-cols-3 gap-x-2 gap-y-1">
          {LOS_GRADES.map((g) => {
            const c = losCounts[g] ?? { ped: 0, veh: 0 }
            return (
              <span key={g} className="flex items-center gap-1">
                <span
                  style={{
                    display: "inline-block",
                    width: 9,
                    height: 9,
                    borderRadius: 9999,
                    backgroundColor: LOS_HEX[g],
                    border: "1px solid #fff",
                  }}
                />
                {g}
                <span className="text-muted-foreground text-[10px]">
                  {c.ped + c.veh > 0 ? `(${c.ped + c.veh})` : ""}
                </span>
              </span>
            )
          })}
        </div>
      </div>
    </div>
  )
}
