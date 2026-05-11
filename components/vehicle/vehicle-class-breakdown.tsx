"use client"

import { Car } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import type { VehicleClassBreakdown as ClassRow } from "@/lib/api"

interface VehicleClassBreakdownProps {
  rows: ClassRow[]
}

const CLASS_COLORS: Record<string, string> = {
  car: "bg-sky-500",
  motorcycle: "bg-fuchsia-500",
  truck: "bg-orange-500",
  bus: "bg-emerald-500",
  van: "bg-amber-500",
  tricycle: "bg-rose-500",
  jeepney: "bg-purple-500",
  bicycle: "bg-teal-500",
}

export function VehicleClassBreakdown({ rows }: VehicleClassBreakdownProps) {
  const total = rows.reduce((acc, r) => acc + r.count, 0)

  return (
    <Card className="h-full flex flex-col rounded-3xl border border-border bg-card shadow-elevated">
      <CardHeader>
        <CardTitle className="text-base">Class breakdown</CardTitle>
        <CardDescription>Detected vehicle classes (PCE-weighted)</CardDescription>
      </CardHeader>
      <CardContent className="flex-1 overflow-auto">
        {rows.length === 0 || total === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center text-muted-foreground">
            <Car className="mb-2 h-6 w-6" />
            <p className="text-xs">No class data yet.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {/* Stacked horizontal bar */}
            <div className="flex h-3 w-full overflow-hidden rounded-full bg-secondary/50">
              {rows.map((row) => {
                const color = CLASS_COLORS[row.className] ?? "bg-slate-500"
                return (
                  <div
                    key={row.className}
                    className={color}
                    style={{ width: `${row.share * 100}%` }}
                    title={`${row.label ?? row.className}: ${row.count}`}
                  />
                )
              })}
            </div>
            {/* Rows */}
            <ul className="space-y-2 text-sm">
              {rows.map((row) => {
                const color = CLASS_COLORS[row.className] ?? "bg-slate-500"
                return (
                  <li key={row.className} className="flex items-center justify-between gap-2">
                    <span className="flex items-center gap-2 text-foreground">
                      <span className={`inline-block h-2.5 w-2.5 rounded-full ${color}`} />
                      <span>{row.label ?? row.className}</span>
                      <span className="text-[10px] text-muted-foreground">×{row.pceMultiplier} PCE</span>
                    </span>
                    <span className="tabular-nums text-muted-foreground">
                      {row.count} <span className="text-[10px]">({(row.share * 100).toFixed(0)}%)</span>
                    </span>
                  </li>
                )
              })}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
