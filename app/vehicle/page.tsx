"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { useRouter } from "next/navigation"
import { AlertCircle, Car, Loader2, ScanLine } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { FootageDatePicker } from "@/components/ui/footage-date-picker"
import { GateGrid } from "@/components/vehicle/gate-grid"
import { VehicleEventFeed } from "@/components/vehicle/vehicle-event-feed"
import {
  getVehicleEvents,
  getVehicleGates,
  getVehicleSummary,
  type EventRecord,
  type VehicleGate,
  type VehicleGateLOS,
  type VehicleSummary,
} from "@/lib/api"

export default function VehicleOverviewPage() {
  const router = useRouter()
  const [detectionMode, setDetectionMode] = useState(true)
  const [selectedDate, setSelectedDate] = useState("")
  const [gates, setGates] = useState<VehicleGate[]>([])
  const [summary, setSummary] = useState<VehicleSummary | null>(null)
  const [events, setEvents] = useState<EventRecord[]>([])
  const [gatesLoading, setGatesLoading] = useState(true)
  const [eventsLoading, setEventsLoading] = useState(true)
  const [pageError, setPageError] = useState<string | null>(null)

  // Gates are seeds — fetch once on mount
  useEffect(() => {
    let cancelled = false
    void getVehicleGates()
      .then((g) => {
        if (!cancelled) setGates(g)
      })
      .catch((error) => {
        if (!cancelled) setPageError(error instanceof Error ? error.message : "Failed to load gates.")
      })
    return () => {
      cancelled = true
    }
  }, [])

  const loadDateScopedData = useCallback(async (date: string) => {
    setGatesLoading(true)
    setEventsLoading(true)
    try {
      const [summaryResponse, eventsResponse] = await Promise.all([
        getVehicleSummary(date || undefined),
        getVehicleEvents(date || undefined),
      ])
      setSummary(summaryResponse)
      setEvents(eventsResponse)
      setPageError(null)
    } catch (error) {
      setPageError(error instanceof Error ? error.message : "Failed to load vehicle data.")
    } finally {
      setGatesLoading(false)
      setEventsLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadDateScopedData(selectedDate)
  }, [loadDateScopedData, selectedDate])

  const losByGate = useMemo<Record<string, VehicleGateLOS>>(() => {
    const map: Record<string, VehicleGateLOS> = {}
    for (const row of summary?.perGateLos ?? []) {
      map[row.gateId] = row
    }
    return map
  }, [summary])

  const totals = useMemo(() => {
    const inGates = gates.filter((g) => g.flowGroup === "In").length
    const outGates = gates.filter((g) => g.flowGroup === "Out").length
    const armedLines = gates.filter((g) => g.detectionLine != null).length
    return { inGates, outGates, armedLines }
  }, [gates])

  return (
    <div className="flex h-full">
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-border bg-card/50 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/15 text-primary">
              <Car className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-white">Vehicle Surveillance Overview</h1>
              <p className="text-xs text-muted-foreground">
                {totals.inGates} entrance · {totals.outGates} exit · {totals.armedLines} armed detection lines
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 flex-wrap justify-end">
            <FootageDatePicker
              value={selectedDate}
              onChange={setSelectedDate}
              highlightedDates={[]}
              allowClear
            />
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-2xl bg-secondary border border-border">
              <ScanLine className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-foreground">Detection</span>
              <Switch
                checked={detectionMode}
                onCheckedChange={setDetectionMode}
                className="data-[state=checked]:bg-accent"
              />
            </div>
            <Button
              variant="outline"
              className="rounded-2xl"
              onClick={() => router.push("/vehicle/dashboard")}
            >
              Open Dashboard
            </Button>
          </div>
        </header>

        {/* Main */}
        <div className="flex-1 overflow-auto p-6">
          {pageError && (
            <div className="mb-4 flex items-start gap-3 rounded-2xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{pageError}</span>
            </div>
          )}

          {summary && (
            <div className="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4">
              <SummaryStat label="Total vehicles" value={summary.totalVehicles.toString()} />
              <SummaryStat label="Gates monitored" value={summary.totalGates.toString()} />
              <SummaryStat
                label="Avg V/C"
                value={summary.averageVcRatio == null ? "—" : summary.averageVcRatio.toFixed(2)}
              />
              <SummaryStat
                label="Dominant LOS"
                value={summary.dominantLos ?? "—"}
                emphasize={summary.dominantLos === "F" || summary.dominantLos === "E"}
              />
            </div>
          )}

          {gatesLoading ? (
            <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground">
              <Loader2 className="mb-4 h-8 w-8 animate-spin" />
              <p className="text-sm">Loading gates…</p>
            </div>
          ) : gates.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-3xl bg-secondary">
                <Car className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="mb-2 text-lg font-medium text-foreground">No gates registered</h3>
              <p className="text-sm text-muted-foreground">
                Vehicle gates ship as fixed seeds. Check the backend logs for a registry warning.
              </p>
            </div>
          ) : (
            <GateGrid
              gates={gates}
              losByGate={losByGate}
              detectionMode={detectionMode}
              onGateClick={() => router.push("/vehicle/dashboard")}
            />
          )}
        </div>
      </div>

      {/* Right sidebar */}
      <aside className="w-80 border-l border-border bg-card/30 flex flex-col h-full">
        <div className="px-4 py-3 border-b border-border">
          <h3 className="text-sm font-medium text-foreground">Gate quick view</h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Live LOS strip. Open the dashboard for the full Leaflet map.
          </p>
        </div>
        <div className="px-4 py-3 border-b border-border space-y-2">
          {(summary?.perGateLos ?? []).map((row) => (
            <div
              key={row.gateId}
              className="flex items-center justify-between rounded-xl border border-border/60 bg-secondary/30 px-3 py-2"
            >
              <div>
                <div className="text-xs font-medium text-foreground">{row.gateName}</div>
                <div className="text-[10px] text-muted-foreground">
                  V/C {row.vcRatio == null ? "—" : row.vcRatio.toFixed(2)} · {row.vehicleCount} veh
                </div>
              </div>
              <div className="text-base font-semibold text-foreground">{row.los ?? "—"}</div>
            </div>
          ))}
        </div>
        <VehicleEventFeed events={events} loading={eventsLoading} />
      </aside>
    </div>
  )
}

function SummaryStat({
  label,
  value,
  emphasize = false,
}: {
  label: string
  value: string
  emphasize?: boolean
}) {
  return (
    <div className="rounded-2xl border border-border bg-secondary/30 p-4">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div
        className={`mt-2 text-2xl font-semibold ${
          emphasize ? "text-rose-300" : "text-foreground"
        }`}
      >
        {value}
      </div>
    </div>
  )
}
