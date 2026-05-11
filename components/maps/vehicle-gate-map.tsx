"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import type {
  CircleMarker as LeafletCircleMarker,
  LatLngBoundsExpression,
  Map as LeafletMap,
} from "leaflet"
import type { VehicleGate, VehicleGateLOS } from "@/lib/api"
import { LOS_GRADES, LOS_HEX, LOS_UNKNOWN_HEX } from "@/lib/los-colors"

const CAMPUS_BOUNDS: LatLngBoundsExpression = [
  [14.63155, 121.07088],
  [14.64807, 121.08418],
]

const TILE_LAYER_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
const TILE_LAYER_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'

interface VehicleGateMapProps {
  gates: VehicleGate[]
  losByGate?: Record<string, VehicleGateLOS>
  selectedGateId?: string | null
  onSelectGate?: (gateId: string) => void
  className?: string
  height?: string
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;")
}

function formatVcRatio(vc: number | null | undefined): string {
  return vc == null || !Number.isFinite(vc) ? "—" : vc.toFixed(2)
}

function buildPopupHtml(gate: VehicleGate, los: VehicleGateLOS | undefined): string {
  const flow = gate.flowGroup === "In" ? "Entrance" : "Exit"
  return `
    <div style="font-family: inherit; min-width: 180px;">
      <div style="font-weight:600; font-size:13px; margin-bottom:4px;">${escapeHtml(gate.name)}</div>
      <div style="font-size:11px; color:#64748b; margin-bottom:6px;">
        ${flow} · ${escapeHtml(gate.countingConfig)}
      </div>
      <div style="display:grid; grid-template-columns: auto auto; gap:4px 12px; font-size:12px;">
        <span style="color:#64748b">LOS</span><span><strong>${los?.los ?? "—"}</strong></span>
        <span style="color:#64748b">V/C</span><span>${formatVcRatio(los?.vcRatio)}</span>
        <span style="color:#64748b">Vehicles</span><span>${los?.vehicleCount ?? 0}</span>
      </div>
    </div>
  `
}

export function VehicleGateMap({
  gates,
  losByGate = {},
  selectedGateId,
  onSelectGate,
  className,
  height = "100%",
}: VehicleGateMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<LeafletMap | null>(null)
  const markerRefs = useRef<Map<string, LeafletCircleMarker>>(new Map())
  const [leaflet, setLeaflet] = useState<typeof import("leaflet") | null>(null)

  // Load leaflet once on the client
  useEffect(() => {
    let cancelled = false
    void import("leaflet").then((mod) => {
      if (cancelled) return
      setLeaflet(mod.default ? (mod.default as typeof import("leaflet")) : (mod as typeof import("leaflet")))
    })
    return () => {
      cancelled = true
    }
  }, [])

  // Initialize map once leaflet is ready
  useEffect(() => {
    if (!leaflet || !containerRef.current || mapRef.current) return
    const map = leaflet.map(containerRef.current, { minZoom: 14 }).fitBounds(CAMPUS_BOUNDS)
    leaflet
      .tileLayer(TILE_LAYER_URL, { attribution: TILE_LAYER_ATTRIBUTION, maxZoom: 19 })
      .addTo(map)
    mapRef.current = map
    return () => {
      map.remove()
      mapRef.current = null
      markerRefs.current.clear()
    }
  }, [leaflet])

  // Sync markers when gates / LOS change
  useEffect(() => {
    const map = mapRef.current
    if (!leaflet || !map) return

    const presentIds = new Set(gates.map((g) => g.id))
    for (const [id, marker] of markerRefs.current.entries()) {
      if (!presentIds.has(id)) {
        marker.remove()
        markerRefs.current.delete(id)
      }
    }

    for (const gate of gates) {
      const los = losByGate[gate.id]
      const fill = los?.los ? LOS_HEX[los.los] : LOS_UNKNOWN_HEX
      const isSelected = selectedGateId === gate.id
      const radius = isSelected ? 14 : 10
      const weight = isSelected ? 3 : 2
      const popupHtml = buildPopupHtml(gate, los)

      const existing = markerRefs.current.get(gate.id)
      if (existing) {
        existing.setLatLng([gate.latitude, gate.longitude])
        existing.setStyle({
          color: "#ffffff",
          fillColor: fill,
          fillOpacity: 0.9,
          weight,
          radius,
        })
        existing.setPopupContent(popupHtml)
        continue
      }

      const marker = leaflet
        .circleMarker([gate.latitude, gate.longitude], {
          radius,
          color: "#ffffff",
          weight,
          fillColor: fill,
          fillOpacity: 0.9,
        })
        .addTo(map)
        .bindPopup(popupHtml)
        .bindTooltip(gate.name, { direction: "top", offset: [0, -10] })

      marker.on("click", () => onSelectGate?.(gate.id))
      markerRefs.current.set(gate.id, marker)
    }
  }, [leaflet, gates, losByGate, selectedGateId, onSelectGate])

  // Pan to selected gate
  useEffect(() => {
    const map = mapRef.current
    if (!map || !selectedGateId) return
    const gate = gates.find((g) => g.id === selectedGateId)
    if (!gate) return
    map.flyTo([gate.latitude, gate.longitude], Math.max(map.getZoom(), 17), { duration: 0.6 })
    markerRefs.current.get(selectedGateId)?.openPopup()
  }, [selectedGateId, gates])

  const losCounts = useMemo(() => {
    const counts: Partial<Record<string, number>> = {}
    for (const row of Object.values(losByGate)) {
      const key = row.los ?? "—"
      counts[key] = (counts[key] ?? 0) + 1
    }
    return counts
  }, [losByGate])

  return (
    <div className={className} style={{ position: "relative", height }}>
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
      <div
        className="absolute right-3 top-3 rounded-xl border border-border/60 bg-card/95 px-3 py-2 text-[11px] text-foreground shadow-elevated-sm"
        style={{ zIndex: 1000 }}
      >
        <div className="mb-1 font-medium">LOS legend</div>
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          {LOS_GRADES.map((g) => {
            const count = losCounts[g] ?? 0
            return (
              <span key={g} className="flex items-center gap-1">
                <span
                  style={{
                    display: "inline-block",
                    width: 10,
                    height: 10,
                    borderRadius: 9999,
                    backgroundColor: LOS_HEX[g],
                    border: "1px solid #fff",
                  }}
                />
                {g}
                {count > 0 && <span className="text-muted-foreground">({count})</span>}
              </span>
            )
          })}
        </div>
      </div>
    </div>
  )
}
