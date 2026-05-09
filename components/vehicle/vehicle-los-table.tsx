"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { VehicleGateLOS } from "@/lib/api"
import { LOS_PILL_CLASS, LOS_PILL_NEUTRAL } from "@/lib/los-colors"

interface VehicleLosTableProps {
  rows: VehicleGateLOS[]
  selectedGateId?: string | null
  onSelect?: (gateId: string) => void
}

export function VehicleLosTable({ rows, selectedGateId, onSelect }: VehicleLosTableProps) {
  return (
    <Card className="bg-secondary/30 border-border/60">
      <CardHeader>
        <CardTitle className="text-base">Per-gate LOS</CardTitle>
        <CardDescription>HCM V/C grade for each monitored gate</CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <div className="divide-y divide-border/60">
          <div className="grid grid-cols-12 gap-2 px-4 py-2 text-[10px] uppercase tracking-wider text-muted-foreground">
            <span className="col-span-3">Gate</span>
            <span className="col-span-2">Flow</span>
            <span className="col-span-2 text-right">Vehicles</span>
            <span className="col-span-2 text-right">V/C</span>
            <span className="col-span-2 text-right">Capacity</span>
            <span className="col-span-1 text-right">LOS</span>
          </div>
          {rows.length === 0 ? (
            <div className="px-4 py-6 text-sm text-muted-foreground">No gate data available.</div>
          ) : (
            rows.map((row) => {
              const isSelected = selectedGateId === row.gateId
              const losClass = row.los ? LOS_PILL_CLASS[row.los] : LOS_PILL_NEUTRAL
              return (
                <button
                  key={row.gateId}
                  type="button"
                  onClick={() => onSelect?.(row.gateId)}
                  className={`grid w-full grid-cols-12 items-center gap-2 px-4 py-2 text-left text-sm transition-colors ${
                    isSelected ? "bg-primary/10" : "hover:bg-secondary/40"
                  }`}
                >
                  <span className="col-span-3 font-medium text-foreground">{row.gateName}</span>
                  <span className="col-span-2">
                    <Badge variant="outline" className="text-[10px]">
                      {row.flowGroup}
                    </Badge>
                  </span>
                  <span className="col-span-2 text-right tabular-nums">{row.vehicleCount}</span>
                  <span className="col-span-2 text-right tabular-nums">
                    {row.vcRatio == null ? "—" : row.vcRatio.toFixed(2)}
                  </span>
                  <span className="col-span-2 text-right tabular-nums text-muted-foreground">
                    {row.capacity.toFixed(0)}
                  </span>
                  <span className="col-span-1 text-right">
                    <span
                      className={`inline-flex h-7 min-w-[1.75rem] items-center justify-center rounded-md border px-2 text-xs font-semibold ${losClass}`}
                    >
                      {row.los ?? "—"}
                    </span>
                  </span>
                </button>
              )
            })
          )}
        </div>
      </CardContent>
    </Card>
  )
}
