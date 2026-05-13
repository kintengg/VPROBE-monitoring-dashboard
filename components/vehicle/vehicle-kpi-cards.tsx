"use client"

import { Activity, ArrowDownToLine, ArrowUpFromLine, Gauge } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import type { VehicleSummary } from "@/lib/api"

interface VehicleKpiCardsProps {
  summary: VehicleSummary | null
}

export function VehicleKpiCards({ summary }: VehicleKpiCardsProps) {
  const inCount = (summary?.perGateLos ?? []).filter((r) => r.flowGroup === "In").reduce((acc, r) => acc + r.vehicleCount, 0)
  const outCount = (summary?.perGateLos ?? []).filter((r) => r.flowGroup === "Out").reduce((acc, r) => acc + r.vehicleCount, 0)
  const avgVc = summary?.averageVcRatio
  // Average LOS = mean V/C across gates → mapped to a letter. Dominant LOS
  // is the worst grade seen on any active gate. Both reflect the selected
  // time window (vehicle_summary is window-aware).
  const averageLos = summary?.averageLos ?? null
  const dominant = summary?.dominantLos

  const cards: Array<{
    label: string
    value: string
    helper?: string
    icon: React.ReactNode
    accent: string
  }> = [
    {
      label: "Total vehicles",
      value: summary ? String(summary.totalVehicles) : "—",
      helper: summary ? `${summary.totalGates} gates monitored` : "—",
      icon: <Activity className="h-4 w-4" />,
      accent: "text-primary",
    },
    {
      label: "Inbound",
      value: String(inCount),
      helper: "Across entrance gates",
      icon: <ArrowDownToLine className="h-4 w-4" />,
      accent: "text-emerald-300",
    },
    {
      label: "Outbound",
      value: String(outCount),
      helper: "Across exit gates",
      icon: <ArrowUpFromLine className="h-4 w-4" />,
      accent: "text-orange-300",
    },
    {
      label: "Average LOS · Worst LOS",
      value: `${averageLos ?? "—"} · ${dominant ?? "—"}`,
      helper: avgVc == null ? "Across the selected window" : `Mean V/C ${avgVc.toFixed(2)}`,
      icon: <Gauge className="h-4 w-4" />,
      accent: dominant === "F" || dominant === "E" ? "text-rose-300" : "text-foreground",
    },
  ]

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
      {cards.map((card) => (
        <Card key={card.label} className="rounded-3xl border border-border bg-card shadow-elevated">
          <CardContent className="p-4">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span className="uppercase tracking-wider">{card.label}</span>
              <span className={card.accent}>{card.icon}</span>
            </div>
            <div className={`mt-2 text-2xl font-semibold ${card.accent}`}>{card.value}</div>
            {card.helper && <div className="mt-1 text-[11px] text-muted-foreground">{card.helper}</div>}
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
