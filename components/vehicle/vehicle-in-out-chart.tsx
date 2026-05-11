"use client"

import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
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
import { Loader2, RotateCcw, ZoomIn } from "lucide-react"
import type { TrafficPoint } from "@/lib/api"

interface VehicleInOutChartProps {
  timeRange: string
  selectedDate: string
  data: TrafficPoint[]
  bucketMinutes: number
  zoomLevel: number
  canZoomIn: boolean
  focusTime?: string | null
  windowStart?: string | null
  windowEnd?: string | null
  loading?: boolean
  onTimeSelect?: (time: string) => void
  onResetZoom?: () => void
  chartType?: "line" | "bar"
  onChartTypeChange?: (value: "line" | "bar") => void
}

const IN_OUT_SERIES = [
  { key: "In", color: "#22C55E" },
  { key: "Out", color: "#EF4444" },
]

function formatRangeLabel(timeRange: string) {
  const labels: Record<string, string> = {
    "whole-day": "whole day",
    "12h": "12 hours",
    "6h": "6 hours",
    "4h": "4 hours",
    "3h": "3 hours",
    "2h": "2 hours",
    "1h": "1 hour",
    "30m": "30 minutes",
  }
  return labels[timeRange] ?? timeRange
}

function formatDateLabel(selectedDate: string) {
  return selectedDate
    ? new Date(selectedDate).toLocaleDateString("en-US", {
        weekday: "long",
        year: "numeric",
        month: "long",
        day: "numeric",
      })
    : "All dates"
}

export function VehicleInOutChart({
  timeRange,
  selectedDate,
  data,
  bucketMinutes,
  zoomLevel,
  canZoomIn,
  focusTime,
  windowStart,
  windowEnd,
  loading = false,
  onTimeSelect,
  onResetZoom,
  chartType = "line",
  onChartTypeChange,
}: VehicleInOutChartProps) {
  const timeLabelsById = new Map(data.map((point) => [point.id, point.time]))
  const subtitle = zoomLevel > 0
    ? `Zoom level ${zoomLevel} · ${windowStart ?? focusTime ?? "--"}–${windowEnd ?? "--"}`
    : `${formatDateLabel(selectedDate)} - ${formatRangeLabel(timeRange)}`

  const handleChartClick = (state: unknown) => {
    if (!canZoomIn || typeof onTimeSelect !== "function") {
      return
    }

    const activePayload = typeof state === "object" && state !== null && "activePayload" in state
      ? (state as { activePayload?: Array<{ payload?: TrafficPoint }> }).activePayload
      : undefined
    const candidate = activePayload?.find((entry) => typeof entry?.payload?.time === "string")?.payload?.time
    if (typeof candidate === "string" && candidate) {
      onTimeSelect(candidate)
    }
  }

  return (
    <div className="rounded-3xl border border-border bg-card p-6 shadow-elevated">
      <div className="mb-6 flex items-center justify-between gap-4">
        <div className="space-y-1">
          <h3 className="text-base font-semibold text-foreground">In And Out Graph</h3>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
          <p className="text-xs text-muted-foreground">
            Cumulative gate flow grouped as In and Out across {bucketMinutes}-minute intervals.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {typeof onChartTypeChange === "function" && (
            <ToggleGroup
              type="single"
              value={chartType}
              onValueChange={(value) => {
                if (value === "line" || value === "bar") {
                  onChartTypeChange(value)
                }
              }}
              variant="outline"
              size="sm"
              className="rounded-xl border border-border"
              aria-label="In And Out chart type"
            >
              <ToggleGroupItem value="line" className="px-3 text-xs">Line</ToggleGroupItem>
              <ToggleGroupItem value="bar" className="px-3 text-xs">Bar</ToggleGroupItem>
            </ToggleGroup>
          )}
          {zoomLevel > 0 && onResetZoom && (
            <Button variant="outline" size="sm" className="rounded-2xl" onClick={onResetZoom}>
              <RotateCcw className="mr-2 h-4 w-4" />
              Reset Zoom
            </Button>
          )}
        </div>
      </div>

      <div className="h-[320px]">
        {loading ? (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            Loading In/Out traffic data...
          </div>
        ) : data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            {chartType === "bar" ? (
              <BarChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 0 }} onClick={handleChartClick}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272A" vertical={false} />
                <XAxis
                  dataKey="id"
                  tickFormatter={(value) => timeLabelsById.get(String(value)) ?? String(value)}
                  tickCount={6}
                  stroke="#71717A"
                  tick={{ fill: "#71717A", fontSize: 12 }}
                  axisLine={{ stroke: "#27272A" }}
                />
                <YAxis stroke="#71717A" tick={{ fill: "#71717A", fontSize: 12 }} axisLine={{ stroke: "#27272A" }} />
                <Tooltip
                  labelFormatter={(value) => timeLabelsById.get(String(value)) ?? String(value)}
                  contentStyle={{ background: "hsl(var(--popover))", border: "1px solid hsl(var(--border))", borderRadius: "12px", fontSize: 12 }}
                  labelStyle={{ color: "hsl(var(--foreground))", fontWeight: 500 }}
                />
                <Legend wrapperStyle={{ paddingTop: "20px" }} formatter={(value) => <span className="text-sm text-foreground">{value}</span>} />
                {IN_OUT_SERIES.map((series) => (
                  <Bar key={series.key} dataKey={series.key} fill={series.color} radius={[4, 4, 0, 0]} />
                ))}
              </BarChart>
            ) : (
              <LineChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 0 }} onClick={handleChartClick}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272A" vertical={false} />
                <XAxis
                  dataKey="id"
                  tickFormatter={(value) => timeLabelsById.get(String(value)) ?? String(value)}
                  tickCount={6}
                  stroke="#71717A"
                  tick={{ fill: "#71717A", fontSize: 12 }}
                  axisLine={{ stroke: "#27272A" }}
                />
                <YAxis stroke="#71717A" tick={{ fill: "#71717A", fontSize: 12 }} axisLine={{ stroke: "#27272A" }} />
                <Tooltip
                  labelFormatter={(value) => timeLabelsById.get(String(value)) ?? String(value)}
                  contentStyle={{ background: "hsl(var(--popover))", border: "1px solid hsl(var(--border))", borderRadius: "12px", fontSize: 12 }}
                  labelStyle={{ color: "hsl(var(--foreground))", fontWeight: 500 }}
                />
                <Legend wrapperStyle={{ paddingTop: "20px" }} formatter={(value) => <span className="text-sm text-foreground">{value}</span>} />
                {IN_OUT_SERIES.map((series) => (
                  <Line
                    key={series.key}
                    type="monotone"
                    dataKey={series.key}
                    stroke={series.color}
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={false}
                  />
                ))}
              </LineChart>
            )}
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-border text-muted-foreground">
            No In/Out traffic data is available for this time range yet.
          </div>
        )}
      </div>

      {canZoomIn && data.length > 0 && (
        <div className="mt-4 flex items-center gap-2 rounded-2xl border border-border/70 bg-secondary/40 px-3 py-2 text-xs text-muted-foreground">
          <ZoomIn className="h-4 w-4" />
          Click a bucket to zoom into a finer time interval for this same range.
        </div>
      )}
    </div>
  )
}
