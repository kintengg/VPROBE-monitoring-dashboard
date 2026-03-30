"use client"

import { Button } from "@/components/ui/button"
import {
  Area,
  AreaChart,
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
import type { LocationTotal, TrafficPoint } from "@/lib/api"

interface PedestrianChartProps {
  title: string
  description: string
  timeRange: string
  selectedDate: string
  data: TrafficPoint[]
  metricKey: "cumulativeUniquePedestrians" | "averageVisiblePedestrians"
  metricLabel: string
  seriesColor: string
  locationTotals?: LocationTotal[]
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

const SERIES_COLORS = ["#22C55E", "#06B6D4", "#3B82F6", "#F59E0B", "#A855F7"]
const RESERVED_SERIES_KEYS = new Set(["time", "cumulativeUniquePedestrians", "averageVisiblePedestrians"])

function formatTimeRangeLabel(timeRange: string) {
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

function formatMetricValue(metricKey: PedestrianChartProps["metricKey"], value: number) {
  return metricKey === "averageVisiblePedestrians" ? value.toFixed(2) : Math.round(value).toLocaleString()
}

const CustomTooltip = ({
  active,
  payload,
  label,
  metricKey,
}: {
  active?: boolean
  payload?: Array<{ name?: string; value?: number | string; color?: string }>
  label?: string
  metricKey: PedestrianChartProps["metricKey"]
}) => {
  const entries = (payload ?? [])
    .filter((entry): entry is { name: string; value: number | string; color?: string } => typeof entry?.name === "string" && entry.value != null)
    .sort((left, right) => Number(right.value ?? 0) - Number(left.value ?? 0))

  if (active && entries.length > 0) {
    return (
      <div className="rounded-2xl border border-border bg-popover p-3 shadow-elevated">
        <p className="mb-2 text-sm font-medium text-foreground">{label}</p>
        <div className="space-y-2">
          {entries.map((entry) => (
            <div key={entry.name} className="flex items-center gap-2 text-sm">
              <div className="h-2 w-2 rounded-full" style={{ backgroundColor: entry.color ?? "#71717A" }} />
              <span className="text-muted-foreground">{entry.name}:</span>
              <span className="font-medium text-foreground">{formatMetricValue(metricKey, Number(entry.value ?? 0))}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return null
}

export function PedestrianChart({
  title,
  description,
  timeRange,
  selectedDate,
  data,
  metricKey,
  metricLabel,
  seriesColor,
  locationTotals = [],
  bucketMinutes,
  zoomLevel,
  canZoomIn,
  focusTime,
  windowStart,
  windowEnd,
  loading = false,
  onTimeSelect,
  onResetZoom,
}: PedestrianChartProps) {
  const locationSeries = Array.from(
    new Set([
      ...locationTotals.map((item) => item.location),
      ...data.flatMap((point) => Object.keys(point).filter((key) => !RESERVED_SERIES_KEYS.has(key))),
    ]),
  ).map((location, index) => ({
    key: location,
    color: SERIES_COLORS[index % SERIES_COLORS.length],
  }))

  const showLocationBreakdown = metricKey === "cumulativeUniquePedestrians" && locationSeries.length > 0

  const totals = locationTotals.map((item, index) => ({
    location: item.location,
    count: item.totalPedestrians,
    color: SERIES_COLORS[index % SERIES_COLORS.length],
  }))

  const subtitle = zoomLevel > 0
    ? `Zoom level ${zoomLevel} · ${windowStart ?? focusTime ?? "--"}–${windowEnd ?? "--"}`
    : `${formatDateLabel(selectedDate)} - ${formatTimeRangeLabel(timeRange)}`

  const handleChartClick = (state: unknown) => {
    if (!canZoomIn || typeof onTimeSelect !== "function") {
      return
    }
    const candidate = typeof state === "object" && state !== null && "activeLabel" in state ? (state as { activeLabel?: unknown }).activeLabel : undefined
    if (typeof candidate === "string" && candidate) {
      onTimeSelect(candidate)
    }
  }

  return (
    <div className="rounded-3xl border border-border bg-card p-6 shadow-elevated">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold text-foreground">{title}</h3>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
          <p className="mt-1 text-xs text-muted-foreground">{description}</p>
        </div>
        {zoomLevel > 0 && onResetZoom && (
          <Button variant="outline" size="sm" className="rounded-2xl" onClick={onResetZoom}>
            <RotateCcw className="mr-2 h-4 w-4" />
            Reset Zoom
          </Button>
        )}
      </div>

      <div className="h-[400px]">
        {loading ? (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            Loading traffic data...
          </div>
        ) : data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            {showLocationBreakdown ? (
              <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }} onClick={handleChartClick}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272A" vertical={false} />
                <XAxis dataKey="time" stroke="#71717A" tick={{ fill: "#71717A", fontSize: 12 }} axisLine={{ stroke: "#27272A" }} />
                <YAxis
                  stroke="#71717A"
                  tick={{ fill: "#71717A", fontSize: 12 }}
                  axisLine={{ stroke: "#27272A" }}
                  label={{ value: metricLabel, angle: -90, position: "insideLeft", fill: "#71717A", fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip metricKey={metricKey} />} />
                <Legend wrapperStyle={{ paddingTop: "20px" }} formatter={(value) => <span className="text-sm text-foreground">{value}</span>} />
                {locationSeries.map((series) => (
                  <Line
                    key={series.key}
                    type="monotone"
                    dataKey={series.key}
                    name={series.key}
                    stroke={series.color}
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 4, fill: series.color }}
                    connectNulls
                  />
                ))}
              </LineChart>
            ) : (
              <AreaChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 0 }} onClick={handleChartClick}>
                <defs>
                  <linearGradient id={`${metricKey}-gradient`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={seriesColor} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={seriesColor} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272A" vertical={false} />
                <XAxis dataKey="time" stroke="#71717A" tick={{ fill: "#71717A", fontSize: 12 }} axisLine={{ stroke: "#27272A" }} />
                <YAxis
                  stroke="#71717A"
                  tick={{ fill: "#71717A", fontSize: 12 }}
                  axisLine={{ stroke: "#27272A" }}
                  label={{ value: metricLabel, angle: -90, position: "insideLeft", fill: "#71717A", fontSize: 12 }}
                />
                <Tooltip content={<CustomTooltip metricKey={metricKey} />} />
                <Legend wrapperStyle={{ paddingTop: "20px" }} formatter={(value) => <span className="text-sm text-foreground">{value}</span>} />
                <Area
                  type="linear"
                  dataKey={metricKey}
                  name={metricLabel}
                  stroke={seriesColor}
                  strokeWidth={2.5}
                  fill={`url(#${metricKey}-gradient)`}
                  dot={false}
                  activeDot={{ r: 4, fill: seriesColor }}
                  cursor={canZoomIn ? "pointer" : "default"}
                />
              </AreaChart>
            )}
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-border text-muted-foreground">
            No pedestrian analytics are available for this time range yet.
          </div>
        )}
      </div>

      {totals.length > 0 && (
        <div className="mt-6 grid grid-cols-2 gap-4 border-t border-border pt-6 md:grid-cols-4">
          {totals.map((item) => (
            <ChartStat key={item.location} location={item.location} count={item.count.toLocaleString()} color={item.color} />
          ))}
        </div>
      )}

      {canZoomIn && data.length > 0 && (
        <div className="mt-4 flex items-center gap-2 rounded-2xl border border-border/70 bg-secondary/40 px-3 py-2 text-xs text-muted-foreground">
          <ZoomIn className="h-4 w-4" />
          Click a bucket to zoom into a finer time interval for this same range.
        </div>
      )}
    </div>
  )
}

function ChartStat({ location, count, color }: { location: string; count: string; color: string }) {
  return (
    <div className="text-center">
      <div className="mb-1 flex items-center justify-center gap-2">
        <div className="h-3 w-3 rounded-full" style={{ backgroundColor: color }} />
        <span className="text-sm text-muted-foreground">{location}</span>
      </div>
      <p className="text-xl font-bold" style={{ color }}>{count}</p>
    </div>
  )
}
