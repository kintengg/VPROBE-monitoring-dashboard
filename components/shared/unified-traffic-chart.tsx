"use client"

import { useMemo, useState } from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { BarChart3, LineChart as LineIcon } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import type { TrafficResponse, VehicleTrafficResponse } from "@/lib/api"

interface UnifiedTrafficChartProps {
  pedestrianTraffic: TrafficResponse | null
  vehicleTraffic: VehicleTrafficResponse | null
  loading?: boolean
}

const PEDESTRIAN_IN_COLOR = "#a855f7" // purple-500
const PEDESTRIAN_OUT_COLOR = "#22d3ee" // cyan-400
const VEHICLE_IN_COLOR = "#34d399" // emerald-400
const VEHICLE_OUT_COLOR = "#fb923c" // orange-400

interface MergedPoint {
  time: string
  pedIn?: number | null
  pedOut?: number | null
  vehIn?: number | null
  vehOut?: number | null
}

function pedSeriesKey(point: Record<string, unknown>, candidates: string[]): number | null {
  for (const key of candidates) {
    const value = point[key]
    if (typeof value === "number" && Number.isFinite(value)) return value
  }
  return null
}

export function UnifiedTrafficChart({
  pedestrianTraffic,
  vehicleTraffic,
  loading,
}: UnifiedTrafficChartProps) {
  const [mode, setMode] = useState<"line" | "bar">("line")

  const merged = useMemo<MergedPoint[]>(() => {
    const byTime = new Map<string, MergedPoint>()
    for (const point of pedestrianTraffic?.series ?? []) {
      const time = String(point.time ?? "")
      if (!time) continue
      const inCount = pedSeriesKey(point as Record<string, unknown>, ["In", "in", "entries", "cumulativeUniquePedestrians"])
      const outCount = pedSeriesKey(point as Record<string, unknown>, ["Out", "out", "exits"])
      const slot = byTime.get(time) ?? { time }
      slot.pedIn = inCount
      slot.pedOut = outCount
      byTime.set(time, slot)
    }
    for (const point of vehicleTraffic?.series ?? []) {
      const time = String(point.time ?? "")
      if (!time) continue
      const slot = byTime.get(time) ?? { time }
      slot.vehIn = typeof point.In === "number" ? point.In : null
      slot.vehOut = typeof point.Out === "number" ? point.Out : null
      byTime.set(time, slot)
    }
    return Array.from(byTime.values()).sort((a, b) => a.time.localeCompare(b.time))
  }, [pedestrianTraffic, vehicleTraffic])

  const isEmpty = !loading && merged.length === 0

  return (
    <Card className="bg-secondary/30 border-border/60">
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-base">In / Out per time bucket</CardTitle>
            <CardDescription>
              Pedestrian and vehicle counts on the same axis
            </CardDescription>
          </div>
          <div className="flex items-center gap-1 rounded-xl border border-border/60 bg-card/40 p-1">
            <Button
              size="sm"
              variant={mode === "line" ? "secondary" : "ghost"}
              className="h-7 gap-1 px-2 text-xs"
              onClick={() => setMode("line")}
            >
              <LineIcon className="h-3.5 w-3.5" /> Line
            </Button>
            <Button
              size="sm"
              variant={mode === "bar" ? "secondary" : "ghost"}
              className="h-7 gap-1 px-2 text-xs"
              onClick={() => setMode("bar")}
            >
              <BarChart3 className="h-3.5 w-3.5" /> Bar
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isEmpty ? (
          <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
            No traffic recorded yet for this date.
          </div>
        ) : (
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              {mode === "line" ? (
                <LineChart data={merged} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                  <XAxis dataKey="time" tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <Tooltip
                    contentStyle={{ background: "#1f2937", border: "1px solid #334155", fontSize: 12 }}
                    labelStyle={{ color: "#cbd5e1" }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="pedIn" name="Ped In" stroke={PEDESTRIAN_IN_COLOR} strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="pedOut" name="Ped Out" stroke={PEDESTRIAN_OUT_COLOR} strokeWidth={2} dot={false} strokeDasharray="4 3" />
                  <Line type="monotone" dataKey="vehIn" name="Veh In" stroke={VEHICLE_IN_COLOR} strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="vehOut" name="Veh Out" stroke={VEHICLE_OUT_COLOR} strokeWidth={2} dot={false} strokeDasharray="4 3" />
                </LineChart>
              ) : (
                <BarChart data={merged} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                  <XAxis dataKey="time" tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <Tooltip
                    contentStyle={{ background: "#1f2937", border: "1px solid #334155", fontSize: 12 }}
                    labelStyle={{ color: "#cbd5e1" }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="pedIn" name="Ped In" fill={PEDESTRIAN_IN_COLOR} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="pedOut" name="Ped Out" fill={PEDESTRIAN_OUT_COLOR} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="vehIn" name="Veh In" fill={VEHICLE_IN_COLOR} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="vehOut" name="Veh Out" fill={VEHICLE_OUT_COLOR} radius={[4, 4, 0, 0]} />
                </BarChart>
              )}
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
