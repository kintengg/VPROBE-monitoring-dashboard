"use client"

import { useMemo } from "react"
import { GateCard } from "./gate-card"
import type { VehicleGate, VehicleGateLOS } from "@/lib/api"

interface GateGridProps {
  gates: VehicleGate[]
  losByGate?: Record<string, VehicleGateLOS>
  detectionMode?: boolean
  onGateClick?: (gate: VehicleGate) => void
}

export function GateGrid({ gates, losByGate = {}, detectionMode, onGateClick }: GateGridProps) {
  const grouped = useMemo(() => {
    const groups: Record<"In" | "Out", VehicleGate[]> = { In: [], Out: [] }
    for (const gate of gates) {
      groups[gate.flowGroup].push(gate)
    }
    return groups
  }, [gates])

  return (
    <div className="space-y-8">
      {(["In", "Out"] as const).map((group) => {
        const list = grouped[group]
        if (list.length === 0) return null
        return (
          <section key={group}>
            <header className="mb-3 flex items-baseline justify-between">
              <h2 className="text-base font-semibold text-foreground">
                {group === "In" ? "Entrance Gates" : "Exit Gates"}
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  ({list.length} {list.length === 1 ? "gate" : "gates"})
                </span>
              </h2>
            </header>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
              {list.map((gate) => (
                <GateCard
                  key={gate.id}
                  gate={gate}
                  losInfo={losByGate[gate.id]}
                  detectionMode={detectionMode}
                  onClick={onGateClick}
                />
              ))}
            </div>
          </section>
        )
      })}
    </div>
  )
}
