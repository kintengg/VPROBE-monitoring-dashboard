"use client"

import dynamic from "next/dynamic"
import { useCallback, useEffect, useMemo, useState } from "react"
import { useRouter } from "next/navigation"
import { AlertCircle, Car, LayoutDashboard, Loader2, RefreshCw, User } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { FootageDatePicker } from "@/components/ui/footage-date-picker"
import { UnifiedTrafficChart } from "@/components/shared/unified-traffic-chart"
import {
  getDashboardOcclusion,
  getDashboardSummary,
  getDashboardTraffic,
  getVehicleGates,
  getVehicleSummary,
  getVehicleTraffic,
  type DashboardSummary,
  type PTSILocation,
  type PTSIMapResponse,
  type TrafficResponse,
  type VehicleGate,
  type VehicleGateLOS,
  type VehicleSummary,
  type VehicleTrafficResponse,
} from "@/lib/api"

const UnifiedLosMap = dynamic(
  () => import("@/components/maps/unified-los-map").then((m) => m.UnifiedLosMap),
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

export default function UnifiedLandingPage() {
  const router = useRouter()
  const [selectedDate, setSelectedDate] = useState("")
  const [vehicleGates, setVehicleGates] = useState<VehicleGate[]>([])
  const [vehicleSummary, setVehicleSummary] = useState<VehicleSummary | null>(null)
  const [vehicleTraffic, setVehicleTraffic] = useState<VehicleTrafficResponse | null>(null)
  const [pedSummary, setPedSummary] = useState<DashboardSummary | null>(null)
  const [pedTraffic, setPedTraffic] = useState<TrafficResponse | null>(null)
  const [pedOcclusion, setPedOcclusion] = useState<PTSIMapResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Gates are static — fetch once
  useEffect(() => {
    let cancelled = false
    void getVehicleGates()
      .then((g) => {
        if (!cancelled) setVehicleGates(g)
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  const refresh = useCallback(async (date: string) => {
    setLoading(true)
    setError(null)
    try {
      const [vSummary, vTraffic, pSummary, pTraffic, pOcclusion] = await Promise.all([
        getVehicleSummary(date || undefined),
        getVehicleTraffic(date || undefined, "whole-day", undefined, 60),
        getDashboardSummary(date || undefined),
        getDashboardTraffic(date || undefined, "whole-day"),
        getDashboardOcclusion(date || undefined, "whole-day"),
      ])
      setVehicleSummary(vSummary)
      setVehicleTraffic(vTraffic)
      setPedSummary(pSummary)
      setPedTraffic(pTraffic)
      setPedOcclusion(pOcclusion)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh(selectedDate)
  }, [refresh, selectedDate])

  const vehicleLosByGate = useMemo<Record<string, VehicleGateLOS>>(() => {
    const map: Record<string, VehicleGateLOS> = {}
    for (const row of vehicleSummary?.perGateLos ?? []) map[row.gateId] = row
    return map
  }, [vehicleSummary])

  const pedestrianLocations: PTSILocation[] = pedOcclusion?.locations ?? []

  const totals = useMemo(() => {
    const pedTotal = pedSummary?.totalUniquePedestrians ?? 0
    const vehTotal = vehicleSummary?.totalVehicles ?? 0
    const dominantPed = inferDominantLos(pedestrianLocations.map((l) => l.los ?? null))
    const dominantVeh = vehicleSummary?.dominantLos ?? null
    return { pedTotal, vehTotal, dominantPed, dominantVeh }
  }, [pedSummary, vehicleSummary, pedestrianLocations])

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between gap-4 border-b border-border bg-card/50 px-6 py-4 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-primary/15 text-primary">
            <LayoutDashboard className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-white">Unified Surveillance Landing</h1>
            <p className="text-xs text-muted-foreground">
              Pedestrian walkways and vehicle gates on a single map, with combined In/Out traffic.
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

        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          <DomainStatCard
            icon={<User className="h-4 w-4" />}
            label="Pedestrians"
            value={String(totals.pedTotal)}
            sub={`Dominant LOS · ${totals.dominantPed ?? "—"}`}
            href="/pedestrian"
            onClick={() => router.push("/pedestrian")}
            tone="ped"
          />
          <DomainStatCard
            icon={<Car className="h-4 w-4" />}
            label="Vehicles"
            value={String(totals.vehTotal)}
            sub={`Dominant LOS · ${totals.dominantVeh ?? "—"}`}
            href="/vehicle"
            onClick={() => router.push("/vehicle")}
            tone="veh"
          />
          <DomainStatCard
            icon={<User className="h-4 w-4" />}
            label="Walkways monitored"
            value={String(pedestrianLocations.length)}
            sub="Triangle pins on the map"
            href="/pedestrian/overview"
            onClick={() => router.push("/pedestrian/overview")}
            tone="ped"
          />
          <DomainStatCard
            icon={<Car className="h-4 w-4" />}
            label="Gates monitored"
            value={String(vehicleGates.length)}
            sub="Circle pins on the map"
            href="/vehicle/overview"
            onClick={() => router.push("/vehicle/overview")}
            tone="veh"
          />
        </div>

        {/* Map row */}
        <div className="overflow-hidden rounded-2xl border border-border/60 bg-secondary/20">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border/60">
            <div>
              <h2 className="text-base font-semibold text-foreground">Campus LOS map</h2>
              <p className="text-xs text-muted-foreground">
                Triangle = pedestrian walkway · Circle = vehicle gate · color = LOS A→F
              </p>
            </div>
          </div>
          <div style={{ height: 480 }}>
            <UnifiedLosMap
              vehicleGates={vehicleGates}
              vehicleLos={vehicleLosByGate}
              pedestrianLocations={pedestrianLocations}
            />
          </div>
        </div>

        {/* Combined traffic chart */}
        <UnifiedTrafficChart
          pedestrianTraffic={pedTraffic}
          vehicleTraffic={vehicleTraffic}
          loading={loading}
        />
      </div>
    </div>
  )
}

function DomainStatCard({
  icon,
  label,
  value,
  sub,
  onClick,
  tone,
}: {
  icon: React.ReactNode
  label: string
  value: string
  sub: string
  href: string
  onClick: () => void
  tone: "ped" | "veh"
}) {
  const accent = tone === "ped" ? "text-fuchsia-300" : "text-emerald-300"
  return (
    <Card
      onClick={onClick}
      className="bg-secondary/30 border-border/60 cursor-pointer transition hover:bg-secondary/50"
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span className="uppercase tracking-wider">{label}</span>
          <span className={accent}>{icon}</span>
        </div>
        <div className="mt-2 text-2xl font-semibold text-foreground">{value}</div>
        <div className="mt-1 text-[11px] text-muted-foreground">{sub}</div>
      </CardContent>
    </Card>
  )
}

function inferDominantLos(grades: (string | null)[]): string | null {
  const order = ["A", "B", "C", "D", "E", "F"]
  let worst = -1
  for (const g of grades) {
    if (!g) continue
    const idx = order.indexOf(g)
    if (idx > worst) worst = idx
  }
  return worst >= 0 ? order[worst] : null
}
