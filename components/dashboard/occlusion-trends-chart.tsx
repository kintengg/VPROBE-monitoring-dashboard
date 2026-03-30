"use client"

import { Button } from "@/components/ui/button"
import {
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

interface OcclusionTrendsChartProps {
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
}

const OCCLUSION_SERIES = [
  { key: "Light", color: "#FACC15" },
  { key: "Moderate", color: "#F97316" },
  { key: "Heavy", color: "#EF4444" },
]

function formatRangeLabel(timeRange: string) {
  return timeRange
    .replace("whole-day", "Whole Day")
    .replace("last-1h", "Last 1 Hour")
    .replace("last-3h", "Last 3 Hours")
    .replace("last-6h", "Last 6 Hours")
    .replace("last-12h", "Last 12 Hours")
    .replace("morning", "Morning")
    .replace("afternoon", "Afternoon")
    .replace("evening", "Evening")
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

export function OcclusionTrendsChart({
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
}: OcclusionTrendsChartProps) {
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
          <h3 className="text-base font-semibold text-foreground">Occlusion Class Trends</h3>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
          <p className="text-xs text-muted-foreground">
            Average visible pedestrians per occlusion class across {bucketMinutes}-minute intervals.
          </p>
        </div>
        {zoomLevel > 0 && onResetZoom && (
          <Button variant="outline" size="sm" className="rounded-2xl" onClick={onResetZoom}>
            <RotateCcw className="mr-2 h-4 w-4" />
            Reset Zoom
          </Button>
        )}
      </div>

      <div className="h-[320px]">
        {loading ? (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            Loading occlusion trends...
          </div>
        ) : data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 0 }} onClick={handleChartClick}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272A" vertical={false} />
              <XAxis
                dataKey="id"
                tickFormatter={(value) => timeLabelsById.get(String(value)) ?? String(value)}
                stroke="#71717A"
                tick={{ fill: "#71717A", fontSize: 12 }}
                axisLine={{ stroke: "#27272A" }}
              />
              <YAxis stroke="#71717A" tick={{ fill: "#71717A", fontSize: 12 }} axisLine={{ stroke: "#27272A" }} />
              <Tooltip labelFormatter={(value) => timeLabelsById.get(String(value)) ?? String(value)} />
              <Legend wrapperStyle={{ paddingTop: "20px" }} formatter={(value) => <span className="text-sm text-foreground">{value}</span>} />
              {OCCLUSION_SERIES.map((series) => (
                <Line
                  key={series.key}
                  type="monotone"
                  dataKey={series.key}
                  stroke={series.color}
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 4, fill: series.color }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-border text-muted-foreground">
            No occlusion-class detections are available for this time range yet.
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