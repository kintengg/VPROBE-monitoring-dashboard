"use client"

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { Loader2 } from "lucide-react"
import type { VehicleGateLOS } from "@/lib/api"

interface VehicleGateLOSChartProps {
  losRows: VehicleGateLOS[]
  selectedDate: string
  loading?: boolean
}

const LOS_COLOR_MAP: Record<string, string> = {
  A: "#22C55E",
  B: "#84CC16",
  C: "#EAB308",
  D: "#F97316",
  E: "#EF4444",
  F: "#B91C1C",
}

const LOS_RANK: Record<string, number> = {
  A: 1, B: 2, C: 3, D: 4, E: 5, F: 6,
}

function formatDateLabel(selectedDate: string) {
  return selectedDate
    ? new Date(selectedDate).toLocaleDateString("en-US", {
        weekday: "long",
        year: "numeric",
        month: "long",
        day: "numeric",
      })
    : "All dates"
}

const CustomTooltip = ({
  active,
  payload,
}: {
  active?: boolean
  payload?: Array<{ payload?: VehicleGateLOS & { losRank: number } }>
}) => {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload
  if (!row) return null
  const losGrade = row.los ?? "--"
  const losColor = LOS_COLOR_MAP[losGrade] ?? "#71717A"
  return (
    <div className="rounded-2xl border border-border bg-popover p-3 shadow-elevated">
      <p className="mb-2 text-sm font-semibold text-foreground">{row.gateName}</p>
      <div className="space-y-1 text-xs">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full" style={{ backgroundColor: losColor }} />
          <span className="text-muted-foreground">LOS:</span>
          <span className="font-bold" style={{ color: losColor }}>{losGrade}</span>
          <span className="text-muted-foreground">({row.losDescription ?? ""})</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">V/C Ratio:</span>
          <span className="font-medium text-foreground">
            {typeof row.vcRatio === "number" ? row.vcRatio.toFixed(3) : "--"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">Volume:</span>
          <span className="font-medium text-foreground">{row.volume ?? "--"} PCE/h</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground">Vehicles:</span>
          <span className="font-medium text-foreground">{row.vehicleCount ?? 0}</span>
        </div>
      </div>
    </div>
  )
}

export function VehicleGateLOSChart({ losRows, selectedDate, loading = false }: VehicleGateLOSChartProps) {
  const chartData = losRows
    .filter((row) => row.los)
    .map((row) => ({
      ...row,
      losRank: LOS_RANK[row.los ?? ""] ?? 0,
    }))
    .sort((a, b) => (b.losRank ?? 0) - (a.losRank ?? 0))

  return (
    <div className="rounded-3xl border border-border bg-card p-6 shadow-elevated">
      <div className="mb-6">
        <h3 className="text-base font-semibold text-foreground">LOS per Gate</h3>
        <p className="text-sm text-muted-foreground">{formatDateLabel(selectedDate)}</p>
        <p className="mt-1 text-xs text-muted-foreground">
          Level of Service grade and V/C ratio for each campus gate.
        </p>
      </div>

      <div className="h-[400px]">
        {loading ? (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            Loading LOS data...
          </div>
        ) : chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 4, right: 48, left: 8, bottom: 4 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#27272A" horizontal={false} />
              <XAxis
                type="number"
                domain={[0, 6]}
                ticks={[1, 2, 3, 4, 5, 6]}
                tickFormatter={(v) => ["", "A", "B", "C", "D", "E", "F"][v] ?? ""}
                stroke="#71717A"
                tick={{ fill: "#71717A", fontSize: 12 }}
                axisLine={{ stroke: "#27272A" }}
              />
              <YAxis
                type="category"
                dataKey="gateName"
                width={72}
                stroke="#71717A"
                tick={{ fill: "#71717A", fontSize: 11 }}
                axisLine={{ stroke: "#27272A" }}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="losRank" radius={[0, 6, 6, 0]} maxBarSize={32}>
                {chartData.map((row) => (
                  <Cell
                    key={row.gateId}
                    fill={LOS_COLOR_MAP[row.los ?? ""] ?? "#71717A"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-border text-muted-foreground">
            No LOS data available for this date yet.
          </div>
        )}
      </div>

      {/* LOS grade legend */}
      {chartData.length > 0 && (
        <div className="mt-5 flex flex-wrap items-center gap-3 border-t border-border pt-4">
          {Object.entries(LOS_COLOR_MAP).map(([grade, color]) => (
            <div key={grade} className="flex items-center gap-1.5 text-xs">
              <div className="h-3 w-3 rounded-sm" style={{ backgroundColor: color }} />
              <span className="text-muted-foreground">
                {grade}&nbsp;—&nbsp;
                {grade === "A" ? "Free Flow" :
                 grade === "B" ? "Stable" :
                 grade === "C" ? "Near Stable" :
                 grade === "D" ? "Approaching" :
                 grade === "E" ? "Unstable" : "Forced/Breakdown"}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
