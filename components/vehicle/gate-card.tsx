"use client"

import { ArrowDownToLine, ArrowUpFromLine, MapPin, ScanLine } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import type { VehicleGate, VehicleGateLOS } from "@/lib/api"
import { LOS_PILL_CLASS, LOS_PILL_NEUTRAL } from "@/lib/los-colors"

interface GateCardProps {
  gate: VehicleGate
  losInfo?: VehicleGateLOS | null
  detectionMode?: boolean
  onClick?: (gate: VehicleGate) => void
}

function formatVcRatio(vc: number | null | undefined) {
  if (vc == null || !Number.isFinite(vc)) return "—"
  return vc.toFixed(2)
}

export function GateCard({ gate, losInfo, detectionMode = true, onClick }: GateCardProps) {
  const FlowIcon = gate.flowGroup === "In" ? ArrowDownToLine : ArrowUpFromLine
  const losClass = losInfo?.los ? LOS_PILL_CLASS[losInfo.los] : LOS_PILL_NEUTRAL

  return (
    <button
      type="button"
      onClick={() => onClick?.(gate)}
      className="group flex flex-col rounded-2xl border border-border bg-secondary/30 p-4 text-left transition-all hover:border-primary/50 hover:bg-secondary/50"
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          <h3 className="text-base font-semibold text-foreground">{gate.name}</h3>
          <p className="mt-0.5 flex items-center gap-1 text-xs text-muted-foreground">
            <MapPin className="h-3 w-3" />
            {gate.latitude.toFixed(5)}, {gate.longitude.toFixed(5)}
          </p>
        </div>
        <Badge variant="outline" className="gap-1">
          <FlowIcon className="h-3 w-3" />
          {gate.flowGroup}
        </Badge>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3 text-center">
        <div className={`rounded-xl border px-2 py-3 ${losClass}`}>
          <div className="text-[10px] uppercase tracking-wider opacity-80">LOS</div>
          <div className="mt-1 text-2xl font-semibold leading-none">{losInfo?.los ?? "—"}</div>
        </div>
        <div className="rounded-xl border border-border/60 bg-card/40 px-2 py-3">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">V/C</div>
          <div className="mt-1 text-lg font-semibold text-foreground leading-none">
            {formatVcRatio(losInfo?.vcRatio)}
          </div>
        </div>
        <div className="rounded-xl border border-border/60 bg-card/40 px-2 py-3">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Vehicles</div>
          <div className="mt-1 text-lg font-semibold text-foreground leading-none">
            {losInfo?.vehicleCount ?? 0}
          </div>
        </div>
      </div>

      <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <ScanLine className="h-3 w-3" />
          {gate.detectionLine ? `${gate.detectionLine.name}` : "no detection line"}
          {detectionMode && gate.detectionLine ? " · armed" : ""}
        </span>
        <span className="font-mono text-[10px] opacity-70">{gate.countingConfig}</span>
      </div>
    </button>
  )
}
