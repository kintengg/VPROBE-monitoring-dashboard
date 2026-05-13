"use client"

import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import {
  Area,
  AreaChart,
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
import type { LocationTotal, TrafficPoint } from "@/lib/api"

interface VehicleAnalyticsChartProps {
  title: string
  description: string
  timeRange: string
  selectedDate: string
  data: TrafficPoint[]
  metricKey: "cumulativeUniquePedestrians" | "averageVisiblePedestrians" | "los"
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
  chartType?: "line" | "bar"
  onChartTypeChange?: (value: "line" | "bar") => void
  legendPosition?: "top" | "bottom"
  useLosLineColors?: boolean
}

import React from "react"

const SERIES_COLORS = ["#22C55E", "#06B6D4", "#3B82F6", "#F59E0B", "#A855F7"]
const RESERVED_SERIES_KEYS = new Set(["id", "time", "cumulativeUniquePedestrians", "averageVisiblePedestrians", "los"])
const LOS_COLOR_MAP: Record<string, string> = {
  A: "#22C55E",
  B: "#84CC16",
  C: "#EAB308",
  D: "#F97316",
  E: "#EF4444",
  F: "#B91C1C",
}
const LOS_SEVERITY_ORDER: Record<string, number> = {
  A: 0,
  B: 1,
  C: 2,
  D: 3,
  E: 4,
  F: 5,
}

const LOS_LABEL_BY_RANK: Record<number, string> = {
  1: "A",
  2: "B",
  3: "C",
  4: "D",
  5: "E",
  6: "F",
}

const LOS_TICKS = [1, 2, 3, 4, 5, 6]

function losYAxisDomainForChart(chartKind: "bar" | "line" | "area") {
  if (chartKind === "bar") {
    return [0, 6]
  }
  return [0.5, 6.5]
}

type LineDotProps = {
  cx?: number
  cy?: number
  payload?: TrafficPoint
  index?: number
}

function lineDotKey(seriesKey: string, variant: "dot" | "activeDot", props: LineDotProps): string {
  const payloadId = props.payload?.id
  const pointId = typeof payloadId === "string" || typeof payloadId === "number" ? String(payloadId) : null
  const pointTime = typeof props.payload?.time === "string" ? props.payload.time : null
  const pointIndex = Number.isFinite(props.index) ? String(props.index) : null
  const cx = Number.isFinite(props.cx) ? String(props.cx) : "na"
  const cy = Number.isFinite(props.cy) ? String(props.cy) : "na"
  const pointIdentity = pointId ?? pointTime ?? pointIndex ?? `${cx}:${cy}`

  return `${seriesKey}:${variant}:${pointIdentity}`
}

function formatTimeRangeLabel(timeRange: string) {
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

function formatMetricValue(metricKey: VehicleAnalyticsChartProps["metricKey"], value: number) {
  if (metricKey === "los") {
    return LOS_LABEL_BY_RANK[Math.round(value)] ?? "--"
  }
  return metricKey === "averageVisiblePedestrians" ? value.toFixed(2) : Math.round(value).toLocaleString()
}

const CustomTooltip = ({
  active,
  payload,
  label,
  metricKey,
}: {
  active?: boolean
  payload?: Array<{ name?: string; value?: number | string; color?: string; payload?: TrafficPoint }>
  label?: string
  metricKey: VehicleAnalyticsChartProps["metricKey"]
}) => {
  const entries = (payload ?? [])
    .filter((entry): entry is { name: string; value: number | string; color?: string; payload?: TrafficPoint } => typeof entry?.name === "string" && entry.value != null)
    .sort((left, right) => Number(right.value ?? 0) - Number(left.value ?? 0))
  const displayLabel = entries[0]?.payload?.time ?? label

  if (active && entries.length > 0) {
    return (
      <div className="rounded-2xl border border-border bg-popover p-3 shadow-elevated">
        <p className="mb-2 text-sm font-medium text-foreground">{displayLabel}</p>
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

export function VehicleAnalyticsChart({
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
  chartType,
  onChartTypeChange,
  legendPosition = "bottom",
  useLosLineColors = true,
}: VehicleAnalyticsChartProps) {
  const timeLabelsById = new Map(data.map((point) => [point.id, point.time]))
  const locationSeries = Array.from(
    new Set([
      ...locationTotals.map((item) => item.location),
      ...data.flatMap((point) =>
        Object.keys(point).filter((key) => !RESERVED_SERIES_KEYS.has(key) && !key.endsWith("__los") && !key.endsWith("__losRank")),
      ),
    ]),
  ).map((location, index) => ({
    key: location,
    color: SERIES_COLORS[index % SERIES_COLORS.length],
  }))

  const lineColorBySeriesKey = Object.fromEntries(
    locationSeries.map((series) => {
      let worstLos: string | null = null
      let worstRank = -1

      for (const point of data) {
        const losValue = point[`${series.key}__los`]
        if (typeof losValue !== "string") {
          continue
        }
        const rank = LOS_SEVERITY_ORDER[losValue]
        if (rank == null) {
          continue
        }
        if (rank > worstRank) {
          worstRank = rank
          worstLos = losValue
        }
      }

      return [series.key, (worstLos && LOS_COLOR_MAP[worstLos]) || series.color]
    }),
  )

  const resolvedLineColorBySeriesKey = useLosLineColors
    ? lineColorBySeriesKey
    : Object.fromEntries(locationSeries.map((series) => [series.key, series.color]))

  const showLocationBreakdown = (metricKey === "cumulativeUniquePedestrians" || metricKey === "los") && locationSeries.length > 0
  const isLosMetric = metricKey === "los"
  const nonNullMetricPointCount = data.reduce((count, point) => {
    const value = point[metricKey]
    return typeof value === "number" && Number.isFinite(value) ? count + 1 : count
  }, 0)
  const shouldShowMetricDots = isLosMetric || nonNullMetricPointCount <= 1
  const chartTypeSelectionEnabled = typeof onChartTypeChange === "function" && typeof chartType === "string"

  const totals = locationTotals.map((item, index) => ({
    location: item.location,
    count: item.totalPedestrians,
    color: SERIES_COLORS[index % SERIES_COLORS.length],
  }))

  const processedData = React.useMemo(() => {
    if (!isLosMetric || !showLocationBreakdown) return data;
    return data.map(point => {
      const newPoint = { ...point };
      for (const series of locationSeries) {
        const losVal = point[`${series.key}__los`];
        if (typeof losVal === "string" && LOS_SEVERITY_ORDER[losVal] !== undefined) {
          newPoint[`${series.key}__losRank`] = LOS_SEVERITY_ORDER[losVal] + 1;
        }
      }
      return newPoint;
    });
  }, [data, isLosMetric, showLocationBreakdown, locationSeries]);

  const subtitle = zoomLevel > 0
    ? `Zoom level ${zoomLevel} · ${windowStart ?? focusTime ?? "--"}–${windowEnd ?? "--"}`
    : `${formatDateLabel(selectedDate)} - ${formatTimeRangeLabel(timeRange)}`

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

  const chartMargin = legendPosition === "top"
    ? { top: 36, right: 30, left: 0, bottom: 0 }
    : { top: 10, right: 30, left: 0, bottom: 0 }

  const legendProps = legendPosition === "top"
    ? {
        verticalAlign: "top" as const,
        align: "center" as const,
        wrapperStyle: { paddingBottom: "8px" },
      }
    : {
        verticalAlign: "bottom" as const,
        align: "center" as const,
        wrapperStyle: { paddingTop: "20px" },
      }

  return (
    <div className="rounded-3xl border border-border bg-card p-6 shadow-elevated">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold text-foreground">{title}</h3>
          <p className="text-sm text-muted-foreground">{subtitle}</p>
          <p className="mt-1 text-xs text-muted-foreground">{description}</p>
        </div>
        <div className="flex items-center gap-2">
          {chartTypeSelectionEnabled && (
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
              aria-label={`${title} chart type`}
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

      <div className="h-[400px]">
        {loading ? (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            Loading traffic data...
          </div>
        ) : data.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            {showLocationBreakdown ? (
              chartType === "bar" ? (
                <BarChart data={processedData} margin={chartMargin} onClick={handleChartClick}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272A" vertical={false} />
                  <XAxis
                    dataKey="id"
                    tickFormatter={(value) => timeLabelsById.get(String(value)) ?? String(value)}
                    tickCount={6}
                    stroke="#71717A"
                    tick={{ fill: "#71717A", fontSize: 12 }}
                    axisLine={{ stroke: "#27272A" }}
                  />
                  <YAxis
                    stroke="#71717A"
                    tick={{ fill: "#71717A", fontSize: 12 }}
                    axisLine={{ stroke: "#27272A" }}
                    domain={isLosMetric ? losYAxisDomainForChart("bar") : undefined}
                    ticks={isLosMetric ? LOS_TICKS : undefined}
                    tickFormatter={isLosMetric ? (value) => LOS_LABEL_BY_RANK[Number(value)] ?? String(value) : undefined}
                    label={{ value: metricLabel, angle: -90, position: "insideLeft", fill: "#71717A", fontSize: 12 }}
                  />
                  <Tooltip content={<CustomTooltip metricKey={metricKey} />} />
                  <Legend {...legendProps} formatter={(value) => <span className="text-sm text-foreground">{value}</span>} />
                  {locationSeries.map((series) => (
                    <Bar key={series.key} dataKey={isLosMetric ? `${series.key}__losRank` : series.key} name={series.key} fill={series.color} radius={[4, 4, 0, 0]} />
                  ))}
                </BarChart>
              ) : (
                <LineChart data={processedData} margin={chartMargin} onClick={handleChartClick}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272A" vertical={false} />
                  <XAxis
                    dataKey="id"
                    tickFormatter={(value) => timeLabelsById.get(String(value)) ?? String(value)}
                    tickCount={6}
                    stroke="#71717A"
                    tick={{ fill: "#71717A", fontSize: 12 }}
                    axisLine={{ stroke: "#27272A" }}
                  />
                  <YAxis
                    stroke="#71717A"
                    tick={{ fill: "#71717A", fontSize: 12 }}
                    axisLine={{ stroke: "#27272A" }}
                    domain={isLosMetric ? losYAxisDomainForChart("line") : undefined}
                    ticks={isLosMetric ? LOS_TICKS : undefined}
                    tickFormatter={isLosMetric ? (value) => LOS_LABEL_BY_RANK[Number(value)] ?? String(value) : undefined}
                    label={{ value: metricLabel, angle: -90, position: "insideLeft", fill: "#71717A", fontSize: 12 }}
                  />
                  <Tooltip content={<CustomTooltip metricKey={metricKey} />} />
                  <Legend {...legendProps} formatter={(value) => <span className="text-sm text-foreground">{value}</span>} />
                  {locationSeries.map((series) => (
                    <Line
                      key={series.key}
                      type="monotone"
                      dataKey={isLosMetric ? `${series.key}__losRank` : series.key}
                      name={series.key}
                      stroke={resolvedLineColorBySeriesKey[series.key] ?? series.color}
                      strokeWidth={2.5}
                      dot={(props: LineDotProps) => {
                        const dataField = isLosMetric ? `${series.key}__losRank` : series.key;
                        const value = props?.payload?.[dataField];
                        const hasRenderableValue =
                          typeof value === "number" &&
                          Number.isFinite(value) &&
                          Number.isFinite(props.cx) &&
                          Number.isFinite(props.cy)

                        if (!hasRenderableValue) {
                          return <></>
                        }

                        const losValue = props?.payload?.[`${series.key}__los`]
                        const pointColor = useLosLineColors && typeof losValue === "string"
                          ? (LOS_COLOR_MAP[losValue] ?? series.color)
                          : (resolvedLineColorBySeriesKey[series.key] ?? series.color)
                        return (
                          <circle
                            key={lineDotKey(series.key, "dot", props)}
                            cx={props.cx}
                            cy={props.cy}
                            r={3}
                            fill={pointColor}
                            stroke={pointColor}
                          />
                        )
                      }}
                      activeDot={false}
                      connectNulls
                    />
                  ))}
                </LineChart>
              )
            ) : (
              chartType === "bar" ? (
                <BarChart data={data} margin={chartMargin} onClick={handleChartClick}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272A" vertical={false} />
                  <XAxis
                    dataKey="id"
                    tickFormatter={(value) => timeLabelsById.get(String(value)) ?? String(value)}
                    tickCount={6}
                    stroke="#71717A"
                    tick={{ fill: "#71717A", fontSize: 12 }}
                    axisLine={{ stroke: "#27272A" }}
                  />
                  <YAxis
                    stroke="#71717A"
                    tick={{ fill: "#71717A", fontSize: 12 }}
                    axisLine={{ stroke: "#27272A" }}
                    domain={isLosMetric ? losYAxisDomainForChart("bar") : undefined}
                    ticks={isLosMetric ? LOS_TICKS : undefined}
                    tickFormatter={isLosMetric ? (value) => LOS_LABEL_BY_RANK[Number(value)] ?? String(value) : undefined}
                    label={{ value: metricLabel, angle: -90, position: "insideLeft", fill: "#71717A", fontSize: 12 }}
                  />
                  <Tooltip content={<CustomTooltip metricKey={metricKey} />} />
                  <Legend {...legendProps} formatter={(value) => <span className="text-sm text-foreground">{value}</span>} />
                  <Bar dataKey={metricKey} name={metricLabel} fill={seriesColor} radius={[4, 4, 0, 0]} cursor={canZoomIn ? "pointer" : "default"} />
                </BarChart>
              ) : chartType === "line" ? (
                <LineChart data={data} margin={chartMargin} onClick={handleChartClick}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272A" vertical={false} />
                  <XAxis
                    dataKey="id"
                    tickFormatter={(value) => timeLabelsById.get(String(value)) ?? String(value)}
                    tickCount={6}
                    stroke="#71717A"
                    tick={{ fill: "#71717A", fontSize: 12 }}
                    axisLine={{ stroke: "#27272A" }}
                  />
                  <YAxis
                    stroke="#71717A"
                    tick={{ fill: "#71717A", fontSize: 12 }}
                    axisLine={{ stroke: "#27272A" }}
                    domain={isLosMetric ? losYAxisDomainForChart("line") : undefined}
                    ticks={isLosMetric ? LOS_TICKS : undefined}
                    tickFormatter={isLosMetric ? (value) => LOS_LABEL_BY_RANK[Number(value)] ?? String(value) : undefined}
                    label={{ value: metricLabel, angle: -90, position: "insideLeft", fill: "#71717A", fontSize: 12 }}
                  />
                  <Tooltip content={<CustomTooltip metricKey={metricKey} />} />
                  <Legend {...legendProps} formatter={(value) => <span className="text-sm text-foreground">{value}</span>} />
                  <Line
                    type="monotone"
                    dataKey={metricKey}
                    name={metricLabel}
                    stroke={seriesColor}
                    strokeWidth={2.5}
                    dot={shouldShowMetricDots ? ((props: LineDotProps) => {
                      const value = props.payload?.[metricKey]
                      const hasRenderableValue =
                        typeof value === "number" &&
                        Number.isFinite(value) &&
                        Number.isFinite(props.cx) &&
                        Number.isFinite(props.cy)

                      return (
                        <circle
                          key={lineDotKey(metricKey, "dot", props)}
                          cx={hasRenderableValue ? props.cx : 0}
                          cy={hasRenderableValue ? props.cy : 0}
                          r={hasRenderableValue ? 3 : 0}
                          fill={seriesColor}
                          stroke={seriesColor}
                        />
                      )
                    }) : false}
                    activeDot={false}
                    cursor={canZoomIn ? "pointer" : "default"}
                  />
                </LineChart>
              ) : (
                <AreaChart data={data} margin={chartMargin} onClick={handleChartClick}>
                  <defs>
                    <linearGradient id={`${metricKey}-gradient`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={seriesColor} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={seriesColor} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272A" vertical={false} />
                  <XAxis
                    dataKey="id"
                    tickFormatter={(value) => timeLabelsById.get(String(value)) ?? String(value)}
                    tickCount={6}
                    stroke="#71717A"
                    tick={{ fill: "#71717A", fontSize: 12 }}
                    axisLine={{ stroke: "#27272A" }}
                  />
                  <YAxis
                    stroke="#71717A"
                    tick={{ fill: "#71717A", fontSize: 12 }}
                    axisLine={{ stroke: "#27272A" }}
                    domain={isLosMetric ? losYAxisDomainForChart("area") : undefined}
                    ticks={isLosMetric ? LOS_TICKS : undefined}
                    tickFormatter={isLosMetric ? (value) => LOS_LABEL_BY_RANK[Number(value)] ?? String(value) : undefined}
                    label={{ value: metricLabel, angle: -90, position: "insideLeft", fill: "#71717A", fontSize: 12 }}
                  />
                  <Tooltip content={<CustomTooltip metricKey={metricKey} />} />
                  <Legend {...legendProps} formatter={(value) => <span className="text-sm text-foreground">{value}</span>} />
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
              )
            )}
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-border text-muted-foreground">
            No vehicle analytics are available for this time range yet.
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

      {isLosMetric && data.length > 0 && (
        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-border pt-4 text-xs text-muted-foreground">
          <span className="font-medium uppercase tracking-wider text-[10px]">LOS</span>
          {(["A", "B", "C", "D", "E", "F"] as const).map((grade) => (
            <span key={grade} className="inline-flex items-center gap-1.5">
              <span
                aria-hidden="true"
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ backgroundColor: LOS_COLOR_MAP[grade] }}
              />
              <span className="text-foreground">{grade}</span>
            </span>
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
