"use client"

import dynamic from "next/dynamic"
import { useCallback, useEffect, useMemo, useState } from "react"
import { AlertCircle, Car, Loader2, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { FootageDatePicker } from "@/components/ui/footage-date-picker"
import { VehicleKpiCards } from "@/components/vehicle/vehicle-kpi-cards"
import { VehicleTrafficChart } from "@/components/vehicle/vehicle-traffic-chart"
import { VehicleLosTable } from "@/components/vehicle/vehicle-los-table"
import { VehicleClassBreakdown } from "@/components/vehicle/vehicle-class-breakdown"
import {
  getVehicleClassBreakdown,
  getVehicleGates,
  getVehicleSummary,
  getVehicleTraffic,
  type VehicleClassBreakdown as ClassRow,
  type VehicleGate,
  type VehicleGateLOS,
  type VehicleSummary,
  type VehicleTrafficResponse,
} from "@/lib/api"

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
  const [selectedDate, setSelectedDate] = useState("")
  const [gates, setGates] = useState<VehicleGate[]>([])
  const [summary, setSummary] = useState<VehicleSummary | null>(null)
  const [classRows, setClassRows] = useState<ClassRow[]>([])
  const [traffic, setTraffic] = useState<VehicleTrafficResponse | null>(null)
  const [selectedGateId, setSelectedGateId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async (date: string) => {
    setLoading(true)
    setError(null)
    try {
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
  }, [])

  // Gates are seeds — fetch once on mount, not on every date change
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

  useEffect(() => {
    void refresh(selectedDate)
  }, [refresh, selectedDate])

  const losRows = summary?.perGateLos ?? []
  const losByGate = useMemo<Record<string, VehicleGateLOS>>(() => {
    const map: Record<string, VehicleGateLOS> = {}
    for (const row of losRows) {
      map[row.gateId] = row
    }
    return map
  }, [losRows])

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
        <div className="flex items-center gap-3">
          <FootageDatePicker
            value={selectedDate}
            onChange={setSelectedDate}
            highlightedDates={[]}
            allowClear
          />
          <Button variant="outline" size="sm" onClick={() => void refresh(selectedDate)} disabled={loading}>
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

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          {/* Map spans 2 cols on xl */}
          <div className="overflow-hidden rounded-2xl border border-border/60 bg-secondary/20 xl:col-span-2">
            <div className="flex items-center justify-between px-4 py-3 border-b border-border/60">
              <div>
                <h2 className="text-base font-semibold text-foreground">Campus gate map</h2>
                <p className="text-xs text-muted-foreground">
                  Click a gate to highlight it in the LOS table below.
                </p>
              </div>
            </div>
            <div style={{ height: 420 }}>
              <VehicleGateMap
                gates={gates}
                losByGate={losByGate}
                selectedGateId={selectedGateId}
                onSelectGate={setSelectedGateId}
              />
            </div>
          </div>

          <div className="xl:col-span-1">
            <VehicleClassBreakdown rows={classRows} />
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          <div className="xl:col-span-2">
            <VehicleTrafficChart data={traffic} loading={loading} />
          </div>
          <div className="xl:col-span-1">
            <VehicleLosTable rows={losRows} selectedGateId={selectedGateId} onSelect={setSelectedGateId} />
          </div>
        </div>
      </div>
    </div>
  )
}
