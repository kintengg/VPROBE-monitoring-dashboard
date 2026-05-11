"use client"

import dynamic from "next/dynamic"
import { useCallback, useEffect, useMemo, useState } from "react"
import { useRouter } from "next/navigation"
import { AlertCircle, Car, Clock, Loader2, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { FootageDatePicker } from "@/components/ui/footage-date-picker"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { VehicleKpiCards } from "@/components/vehicle/vehicle-kpi-cards"
import { VehicleInOutChart } from "@/components/vehicle/vehicle-in-out-chart"
import { VehicleClassBreakdown } from "@/components/vehicle/vehicle-class-breakdown"
import { VehicleAnalyticsChart } from "@/components/vehicle/vehicle-analytics-chart"
import {
  getDashboardLOS,
  getDashboardTraffic,
  getDashboardTrafficByLocation,
  getLocations,
  getVehicleClassBreakdown,
  getVehicleGates,
  getVehicleSummary,
  getVehicleTraffic,
  type TrafficByLocationResponse,
  type TrafficPoint,
  type TrafficResponse,
  type VehicleClassBreakdown as ClassRow,
  type VehicleGate,
  type VehicleGateLOS,
  type VehicleSummary,
  type VehicleTrafficResponse,
} from "@/lib/api"

const TIME_RANGE_OPTIONS = [
  { value: "whole-day", label: "Whole day (24h)" },
  { value: "12h", label: "12 hours" },
  { value: "6h", label: "6 hours" },
  { value: "4h", label: "4 hours" },
  { value: "3h", label: "3 hours" },
  { value: "2h", label: "2 hours" },
  { value: "1h", label: "1 hour" },
  { value: "30m", label: "30 minutes" },
] as const

const START_TIME_OPTIONS = Array.from({ length: 48 }, (_, index) => {
  const totalMinutes = index * 30
  const hour24 = Math.floor(totalMinutes / 60)
  const hours = String(hour24).padStart(2, "0")
  const minutes = String(totalMinutes % 60).padStart(2, "0")
  const value = `${hours}:${minutes}`
  const suffix = hour24 >= 12 ? "PM" : "AM"
  const hour12 = hour24 % 12 || 12
  const label = `${hour12}:${minutes} ${suffix}`
  return { value, label }
})

const VehicleGateMap = dynamic(
  () => import("@/components/maps/vehicle-gate-map").then((m) => m.VehicleGateMap),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading map…
      </div>
    ),
  },
)

export default function VehicleDashboardPage() {
  const router = useRouter()

  // Date / time window controls
  const [selectedDate, setSelectedDate] = useState("")
  const [timeRange, setTimeRange] = useState("whole-day")
  const [startTime, setStartTime] = useState("06:00")
  const [focusTime, setFocusTime] = useState<string | undefined>(undefined)
  const [zoomLevel, setZoomLevel] = useState(0)

  // Chart type toggles
  const [vehicleChartType, setVehicleChartType] = useState<"line" | "bar">("line")
  const [losChartType, setLosChartType] = useState<"line" | "bar">("bar")
  const [allGatesVehicleChartType, setAllGatesVehicleChartType] = useState<"line" | "bar">("line")
  const [inOutChartType, setInOutChartType] = useState<"line" | "bar">("line")

  // Gate data (static)
  const [gates, setGates] = useState<VehicleGate[]>([])
  const [selectedGateId, setSelectedGateId] = useState<string | null>(null)

  // Summary / classification / In-Out traffic
  const [summary, setSummary] = useState<VehicleSummary | null>(null)
  const [classRows, setClassRows] = useState<ClassRow[]>([])
  const [traffic, setTraffic] = useState<VehicleTrafficResponse | null>(null)
  const [availableDates, setAvailableDates] = useState<string[]>([])
  const [autoDateSet, setAutoDateSet] = useState(false)

  // Analytics charts (new — from dashboard analytics endpoints)
  const [vehicleCountTraffic, setVehicleCountTraffic] = useState<TrafficResponse | null>(null)
  const [losTraffic, setLosTraffic] = useState<TrafficResponse | null>(null)
  const [trafficByLocation, setTrafficByLocation] = useState<TrafficByLocationResponse | null>(null)

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async (
    date: string,
    range: string,
    start: string,
    focus: string | undefined,
    zoom: number,
    gateId: string | null,
  ) => {
    setLoading(true)
    setError(null)
    try {
      // Core vehicle data — errors are surfaced as a page banner.
      const [summaryResponse, classResponse, trafficResponse] = await Promise.all([
        getVehicleSummary(date || undefined),
        getVehicleClassBreakdown(date || undefined),
        getVehicleTraffic(date || undefined, "whole-day", 60),
      ])
      setSummary(summaryResponse)
      setClassRows(classResponse)
      setTraffic(trafficResponse)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }

    // Analytics endpoints are optional — a 404 or any other error is swallowed
    // so an unimplemented/missing backend route never blocks the page or shows a banner.
    const [countResult, losResult, byLocationResult] = await Promise.allSettled([
      getDashboardTraffic(date || undefined, range, focus, zoom, start, gateId ?? undefined),
      getDashboardLOS(date || undefined, range, focus, zoom, gateId ?? undefined, start),
      getDashboardTrafficByLocation(date || undefined, range, focus, zoom, start),
    ])

    setVehicleCountTraffic(countResult.status === "fulfilled" ? countResult.value : null)
    setLosTraffic(losResult.status === "fulfilled" ? losResult.value : null)
    setTrafficByLocation(byLocationResult.status === "fulfilled" ? byLocationResult.value : null)
  }, [])

  // Gates are static — fetch once on mount
  useEffect(() => {
    let cancelled = false
    void getVehicleGates()
      .then((g) => {
        if (!cancelled) setGates(g)
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  // Fetch all available dates for the date picker + auto-select latest
  useEffect(() => {
    let cancelled = false
    void getLocations({ domain: "vehicle" })
      .then((locations) => {
        if (cancelled) return
        const allDates = Array.from(new Set(locations.flatMap((l) => l.videos.map((v) => v.date)))).sort()
        setAvailableDates(allDates)
        
        if (!autoDateSet && allDates.length > 0) {
          const latest = allDates.at(-1)
          if (latest) {
            setSelectedDate(latest)
            setAutoDateSet(true)
          }
        }
      })
      .catch((e) => {
        console.error("Failed to load vehicle dates:", e)
      })
    return () => {
      cancelled = true
    }
  }, [autoDateSet])

  useEffect(() => {
    void refresh(selectedDate, timeRange, startTime, focusTime, zoomLevel, selectedGateId)
  }, [refresh, selectedDate, timeRange, startTime, focusTime, zoomLevel, selectedGateId])

  const losRows = summary?.perGateLos ?? []
  const losByGate = useMemo<Record<string, VehicleGateLOS>>(() => {
    const map: Record<string, VehicleGateLOS> = {}
    for (const row of losRows) {
      map[row.gateId] = row
    }
    return map
  }, [losRows])

  const selectedGateName = useMemo(
    () => gates.find((g) => g.id === selectedGateId)?.name ?? "Selected gate",
    [gates, selectedGateId],
  )

  // Time-window handlers — reset zoom whenever the window changes
  const handleTimeRangeChange = (value: string) => {
    setTimeRange(value)
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  const handleStartTimeChange = (value: string) => {
    setStartTime(value)
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  const handleDateChange = (value: string) => {
    setSelectedDate(value)
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  const handleAnalyticsZoom = (time: string) => {
    const canZoom =
      Boolean(vehicleCountTraffic?.canZoomIn) ||
      Boolean(losTraffic?.canZoomIn) ||
      Boolean(trafficByLocation?.canZoomIn)
    if (!canZoom) return
    setFocusTime(time)
    setZoomLevel((current) => current + 1)
  }

  const handleResetZoom = () => {
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between gap-4 border-b border-border bg-card/50 px-6 py-4 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/15 text-primary">
            <Car className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-white">Vehicle Dashboard</h1>
            <p className="text-xs text-muted-foreground">
              Gate-level LOS, V/C, and class breakdown sourced from the vehicle pipeline.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3 flex-wrap justify-end">
          <FootageDatePicker
            value={selectedDate}
            onChange={handleDateChange}
            highlightedDates={availableDates}
            allowClear
          />

          {/* Time range selector */}
          <Select value={timeRange} onValueChange={handleTimeRangeChange}>
            <SelectTrigger className="h-9 w-44 rounded-2xl border-border bg-secondary text-foreground">
              <Clock className="mr-2 h-4 w-4 text-muted-foreground shrink-0" />
              <SelectValue placeholder="Time range" />
            </SelectTrigger>
            <SelectContent className="rounded-xl border-border bg-popover">
              {TIME_RANGE_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value} className="rounded-lg text-foreground">
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Start time selector */}
          <Select value={startTime} onValueChange={handleStartTimeChange}>
            <SelectTrigger className="h-9 w-36 rounded-2xl border-border bg-secondary text-foreground">
              <SelectValue placeholder="Start time" />
            </SelectTrigger>
            <SelectContent className="max-h-80 rounded-xl border-border bg-popover">
              {START_TIME_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value} className="rounded-lg text-foreground">
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button variant="outline" className="rounded-2xl" onClick={() => router.push("/vehicle/overview")}>
            Open Overview
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => void refresh(selectedDate, timeRange, startTime, focusTime, zoomLevel, selectedGateId)}
            disabled={loading}
          >
            {loading ? (
              <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="mr-2 h-3.5 w-3.5" />
            )}
            Refresh
          </Button>
        </div>
      </header>

      <div className="flex-1 overflow-auto px-6 py-6 space-y-6">
        {error && (
          <div className="flex items-start gap-2 rounded-2xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <VehicleKpiCards summary={summary} />

        {/* Map + class breakdown row */}
        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          {/* Map spans 2 cols on xl */}
          <div className="overflow-hidden rounded-3xl border border-border bg-card shadow-elevated xl:col-span-2" style={{ minHeight: 490 }}>
            <div className="flex items-center justify-between px-4 py-3 border-b border-border/60">
              <div>
                <h2 className="text-base font-semibold text-foreground">Campus gate map</h2>
                <p className="text-xs text-muted-foreground">
                  Click a gate to highlight it in the analytics charts below.
                </p>
              </div>
            </div>
            <div style={{ height: 450 }}>
              <VehicleGateMap
                gates={gates}
                losByGate={losByGate}
                selectedGateId={selectedGateId}
                onSelectGate={setSelectedGateId}
              />
            </div>
          </div>

          <div className="xl:col-span-1" style={{ minHeight: 490 }}>
            <VehicleClassBreakdown rows={classRows} />
          </div>
        </div>

        {/* ── Analytics charts (ported from surveillance-system dashboard) ── */}
        <div className="space-y-6">
          {/* Row 1: per-gate vehicle count + per-gate LOS */}
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
            <VehicleAnalyticsChart
              title={selectedGateId ? `Vehicle Count – ${selectedGateName}` : "Vehicle Count"}
              description={
                selectedGateId
                  ? `Cumulative vehicle count for ${selectedGateName} over the selected time window.`
                  : "Select a gate on the map to view its vehicle count trend."
              }
              timeRange={timeRange}
              selectedDate={selectedDate}
              data={vehicleCountTraffic?.series ?? []}
              metricKey="cumulativeUniquePedestrians"
              metricLabel="Vehicle Count"
              seriesColor="#22C55E"
              bucketMinutes={vehicleCountTraffic?.bucketMinutes ?? 60}
              zoomLevel={vehicleCountTraffic?.zoomLevel ?? 0}
              canZoomIn={vehicleCountTraffic?.canZoomIn ?? false}
              focusTime={vehicleCountTraffic?.focusTime}
              windowStart={vehicleCountTraffic?.windowStart}
              windowEnd={vehicleCountTraffic?.windowEnd}
              loading={loading}
              onTimeSelect={handleAnalyticsZoom}
              onResetZoom={handleResetZoom}
              chartType={vehicleChartType}
              onChartTypeChange={setVehicleChartType}
            />

            <VehicleAnalyticsChart
              title={selectedGateId ? `LOS – ${selectedGateName}` : "LOS"}
              description={
                selectedGateId
                  ? `Level of Service trend for ${selectedGateName} across the selected time window.`
                  : "Select a gate from the map above to view its Level of Service trend."
              }
              timeRange={timeRange}
              selectedDate={selectedDate}
              data={losTraffic?.series ?? []}
              metricKey="los"
              metricLabel="LOS"
              seriesColor="#06B6D4"
              bucketMinutes={losTraffic?.bucketMinutes ?? 60}
              zoomLevel={losTraffic?.zoomLevel ?? 0}
              canZoomIn={losTraffic?.canZoomIn ?? false}
              focusTime={losTraffic?.focusTime}
              windowStart={losTraffic?.windowStart}
              windowEnd={losTraffic?.windowEnd}
              loading={loading}
              onTimeSelect={handleAnalyticsZoom}
              onResetZoom={handleResetZoom}
              chartType={losChartType}
              onChartTypeChange={setLosChartType}
            />
          </div>

          {/* Row 2: all-gates vehicle count breakdown + all-gates LOS breakdown */}
          <VehicleAnalyticsChart
            title="Vehicle Count (All Gates)"
            description="Gate-by-gate cumulative vehicle count for the selected date and time window."
            timeRange={timeRange}
            selectedDate={selectedDate}
            data={trafficByLocation?.series ?? []}
            metricKey="cumulativeUniquePedestrians"
            metricLabel="Vehicle Count"
            seriesColor="#22C55E"
            bucketMinutes={trafficByLocation?.bucketMinutes ?? 60}
            zoomLevel={trafficByLocation?.zoomLevel ?? 0}
            canZoomIn={trafficByLocation?.canZoomIn ?? false}
            focusTime={trafficByLocation?.focusTime}
            windowStart={trafficByLocation?.windowStart}
            windowEnd={trafficByLocation?.windowEnd}
            loading={loading}
            onTimeSelect={handleAnalyticsZoom}
            onResetZoom={handleResetZoom}
            chartType={allGatesVehicleChartType}
            onChartTypeChange={setAllGatesVehicleChartType}
            legendPosition="top"
            useLosLineColors={false}
          />

          <VehicleAnalyticsChart
            title="LOS (All Gates)"
            description="Gate-by-gate Level of Service trend for the selected date and time window."
            timeRange={timeRange}
            selectedDate={selectedDate}
            data={trafficByLocation?.series ?? []}
            metricKey="los"
            metricLabel="LOS"
            seriesColor="#06B6D4"
            bucketMinutes={trafficByLocation?.bucketMinutes ?? 60}
            zoomLevel={trafficByLocation?.zoomLevel ?? 0}
            canZoomIn={trafficByLocation?.canZoomIn ?? false}
            focusTime={trafficByLocation?.focusTime}
            windowStart={trafficByLocation?.windowStart}
            windowEnd={trafficByLocation?.windowEnd}
            loading={loading}
            onTimeSelect={handleAnalyticsZoom}
            onResetZoom={handleResetZoom}
            chartType="bar"
            legendPosition="top"
          />
        </div>

        {/* In/Out traffic chart — full width */}
        <VehicleInOutChart
          timeRange={timeRange}
          selectedDate={selectedDate}
          data={(traffic?.series ?? []) as TrafficPoint[]}
          bucketMinutes={traffic?.bucketMinutes ?? 60}
          zoomLevel={0}
          canZoomIn={false}
          loading={loading}
          chartType={inOutChartType}
          onChartTypeChange={setInOutChartType}
        />
      </div>
    </div>
  )
}
