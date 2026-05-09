"use client"

import { useState } from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { BarChart3, LineChart as LineIcon } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import type { VehicleTrafficResponse } from "@/lib/api"

interface VehicleTrafficChartProps {
  data: VehicleTrafficResponse | null
  loading?: boolean
}

const COLOR_IN = "#34d399" // emerald-400
const COLOR_OUT = "#fb923c" // orange-400

export function VehicleTrafficChart({ data, loading }: VehicleTrafficChartProps) {
  const [mode, setMode] = useState<"line" | "bar">("line")

  const series = data?.series ?? []
  const isEmpty = !loading && series.length === 0

  return (
    <Card className="bg-secondary/30 border-border/60">
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-base">Inbound vs Outbound</CardTitle>
            <CardDescription>
              Vehicle counts per {data?.bucketMinutes ?? 60}-minute bucket
            </CardDescription>
          </div>
          <div className="flex items-center gap-1 rounded-xl border border-border/60 bg-card/40 p-1">
            <Button
              size="sm"
              variant={mode === "line" ? "secondary" : "ghost"}
              className="h-7 gap-1 px-2 text-xs"
              onClick={() => setMode("line")}
            >
              <LineIcon className="h-3.5 w-3.5" /> Line
            </Button>
            <Button
              size="sm"
              variant={mode === "bar" ? "secondary" : "ghost"}
              className="h-7 gap-1 px-2 text-xs"
              onClick={() => setMode("bar")}
            >
              <BarChart3 className="h-3.5 w-3.5" /> Bar
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isEmpty ? (
          <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
            No vehicle traffic recorded yet for this date.
          </div>
        ) : (
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              {mode === "line" ? (
                <LineChart data={series} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                  <XAxis dataKey="time" tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <Tooltip
                    contentStyle={{ background: "#1f2937", border: "1px solid #334155", fontSize: 12 }}
                    labelStyle={{ color: "#cbd5e1" }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Line type="monotone" dataKey="In" stroke={COLOR_IN} strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="Out" stroke={COLOR_OUT} strokeWidth={2} dot={false} />
                </LineChart>
              ) : (
                <BarChart data={series} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
                  <XAxis dataKey="time" tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <Tooltip
                    contentStyle={{ background: "#1f2937", border: "1px solid #334155", fontSize: 12 }}
                    labelStyle={{ color: "#cbd5e1" }}
                  />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="In" fill={COLOR_IN} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="Out" fill={COLOR_OUT} radius={[4, 4, 0, 0]} />
                </BarChart>
              )}
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
