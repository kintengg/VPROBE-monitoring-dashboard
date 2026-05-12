"use client"

import { Car } from "lucide-react"
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

const CLASS_HEX: Record<string, string> = {
  car: "#0EA5E9",
  motorcycle: "#D946EF",
  truck: "#F97316",
  bus: "#10B981",
  van: "#F59E0B",
  tricycle: "#F43F5E",
  jeepney: "#A855F7",
  bicycle: "#14B8A6",
}

export function VehicleClassBreakdown({ rows }: VehicleClassBreakdownProps) {
  const total = rows.reduce((acc, r) => acc + r.count, 0)

  return (
    <div className="h-full rounded-3xl border border-border bg-card p-6 shadow-elevated flex flex-col">
      <div className="mb-5">
        <h3 className="text-base font-semibold text-foreground">Class Breakdown</h3>
        <p className="text-sm text-muted-foreground">Detected vehicle classes for this date</p>
      </div>

      {rows.length === 0 || total === 0 ? (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 text-muted-foreground">
          <Car className="h-8 w-8 opacity-40" />
          <p className="text-sm">No class data yet.</p>
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          {/* Stacked bar */}
          <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-secondary/50">
            {rows.map((row) => (
              <div
                key={row.className}
                className={CLASS_COLORS[row.className] ?? "bg-slate-500"}
                style={{ width: `${row.share * 100}%` }}
                title={`${row.label ?? row.className}: ${row.count} (${(row.share * 100).toFixed(0)}%)`}
              />
            ))}
          </div>

          {/* Class rows */}
          <ul className="space-y-3">
            {rows.map((row) => {
              const hex = CLASS_HEX[row.className] ?? "#71717A"
              const pct = (row.share * 100).toFixed(0)
              return (
                <li key={row.className} className="flex items-center gap-3">
                  {/* Color dot */}
                  <div
                    className="h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: hex }}
                  />
                  {/* Label */}
                  <span className="flex-1 text-sm text-foreground">
                    {row.label ?? row.className}
                  </span>
                  {/* Progress bar */}
                  <div className="h-1.5 w-24 overflow-hidden rounded-full bg-secondary/60">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${row.share * 100}%`, backgroundColor: hex }}
                    />
                  </div>
                  {/* Count + % */}
                  <span className="w-20 text-right text-sm tabular-nums text-muted-foreground">
                    {row.count.toLocaleString()}{" "}
                    <span className="text-xs opacity-70">({pct}%)</span>
                  </span>
                </li>
              )
            })}
          </ul>

          {/* Total */}
          <div className="mt-2 flex items-center justify-between border-t border-border pt-3 text-sm">
            <span className="text-muted-foreground">Total vehicles</span>
            <span className="font-semibold tabular-nums text-foreground">{total.toLocaleString()}</span>
          </div>
        </div>
      )}
    </div>
  )
}
