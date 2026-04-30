"use client"

import {
  Bar,
  BarChart,
  Line,
  LineChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ComposedChart, Cell
} from "recharts"
import { Loader2 } from "lucide-react"

export function VehicleKPICards({ summary, loading }: { summary: any, loading: boolean }) {
  if (loading || !summary) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="flex h-[120px] items-center justify-center rounded-2xl border border-border bg-card/50">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ))}
      </div>
    )
  }

  const kpis = [
    { title: "Daily Average", value: `${summary.dailyAverage ?? 0} veh/hr` },
    { title: "Peak Volume", value: summary.peakVolume ?? 0 },
    { title: "Peak Hour", value: summary.peakHour ?? "N/A" },
    { title: "Daily Total", value: summary.dailyTotal ?? 0 },
  ]

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {kpis.map((kpi, i) => (
        <div key={i} className="rounded-2xl border border-border bg-card p-6 shadow-sm">
          <p className="text-sm font-medium text-muted-foreground">{kpi.title}</p>
          <div className="mt-2 text-3xl font-bold text-foreground">{kpi.value}</div>
        </div>
      ))}
    </div>
  )
}

export function VehicleCountChart({ data, loading, type }: { data: any[], loading: boolean, type: 'bar' | 'line' }) {
  if (loading) {
    return (
      <div className="flex h-[400px] items-center justify-center rounded-2xl border border-border bg-card/50">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  const chartData = data.map(d => ({
    time: d.time,
    total: d.total,
    ...d.classes
  }))

  const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#0088FE", "#00C49F"]
  const classNames = Array.from(new Set(data.flatMap(d => Object.keys(d.classes))))

  return (
    <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
      <h3 className="mb-6 text-lg font-semibold text-foreground">Daily Vehicle Count</h3>
      <div className="h-[350px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          {type === 'bar' ? (
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
              <XAxis dataKey="time" tick={{ fill: "#A1A1AA", fontSize: 12 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: "#A1A1AA", fontSize: 12 }} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ borderRadius: '12px', border: '1px solid hsl(var(--border))', backgroundColor: 'hsl(var(--card))', color: 'hsl(var(--foreground))' }}
                formatter={(value: number, name: string) => [value, name]}
              />
              <Legend />
              {classNames.map((name, i) => (
                <Bar key={name} dataKey={name} stackId="a" fill={colors[i % colors.length]} />
              ))}
            </BarChart>
          ) : (
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
              <XAxis dataKey="time" tick={{ fill: "#A1A1AA", fontSize: 12 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: "#A1A1AA", fontSize: 12 }} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ borderRadius: '12px', border: '1px solid hsl(var(--border))', backgroundColor: 'hsl(var(--card))', color: 'hsl(var(--foreground))' }}
                formatter={(value: number, name: string, props: any) => {
                  if (name === 'total') return [value, 'Total'];
                  return [value, name];
                }}
              />
              <Legend />
              <Line type="linear" dataKey="total" stroke="#22C55E" strokeWidth={2} dot={true} />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export function VehicleLOSChart({ data, loading, type }: { data: any[], loading: boolean, type: 'bar' | 'line' }) {
  if (loading) {
    return (
      <div className="flex h-[400px] items-center justify-center rounded-2xl border border-border bg-card/50">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  const getLOSColor = (los: string) => {
    switch(los) {
      case 'A': return '#22C55E' // green
      case 'B': return '#84CC16'
      case 'C': return '#EAB308' // yellow
      case 'D': return '#F59E0B' // orange
      case 'E': return '#EF4444' // red
      case 'F': return '#B91C1C' // dark red
      default: return '#22C55E'
    }
  }

  return (
    <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
      <h3 className="mb-6 text-lg font-semibold text-foreground">Level of Service (LOS)</h3>
      <div className="h-[350px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          {type === 'bar' ? (
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
              <XAxis dataKey="time" tick={{ fill: "#A1A1AA", fontSize: 12 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: "#A1A1AA", fontSize: 12 }} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ borderRadius: '12px', border: '1px solid hsl(var(--border))', backgroundColor: 'hsl(var(--card))', color: 'hsl(var(--foreground))' }}
                formatter={(value: number, name: string, props: any) => [`${value} (LOS ${props.payload.los_level})`, 'V/C Ratio']}
              />
              <Bar dataKey="los_vc_ratio" fill="#8884d8">
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getLOSColor(entry.los_level)} />
                ))}
              </Bar>
            </BarChart>
          ) : (
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
              <XAxis dataKey="time" tick={{ fill: "#A1A1AA", fontSize: 12 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: "#A1A1AA", fontSize: 12 }} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ borderRadius: '12px', border: '1px solid hsl(var(--border))', backgroundColor: 'hsl(var(--card))', color: 'hsl(var(--foreground))' }}
                formatter={(value: number, name: string, props: any) => [`${value} (LOS ${props.payload.los_level})`, 'V/C Ratio']}
              />
              <Line type="linear" dataKey="los_vc_ratio" stroke="#EAB308" strokeWidth={2} dot={true} />
            </LineChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export function VehicleAllGatesChart({ data, loading }: { data: any[], loading: boolean }) {
  if (loading) {
    return (
      <div className="flex h-[400px] items-center justify-center rounded-2xl border border-border bg-card/50">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  const chartData = data.map(d => ({
    time: d.time,
    ...d.gates
  }))

  const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#0088FE", "#00C49F"]
  const gateNames = Array.from(new Set(data.flatMap(d => Object.keys(d.gates))))

  return (
    <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
      <h3 className="mb-6 text-lg font-semibold text-foreground">All Gates Vehicle Count</h3>
      <div className="h-[350px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
            <XAxis dataKey="time" tick={{ fill: "#A1A1AA", fontSize: 12 }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fill: "#A1A1AA", fontSize: 12 }} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{ borderRadius: '12px', border: '1px solid hsl(var(--border))', backgroundColor: 'hsl(var(--card))', color: 'hsl(var(--foreground))' }}
            />
            <Legend />
            {gateNames.map((name, i) => (
              <Bar key={name} dataKey={name} fill={colors[i % colors.length]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export function VehicleCampusCountChart({ data, loading }: { data: any[], loading: boolean }) {
  if (loading) {
    return (
      <div className="flex h-[400px] items-center justify-center rounded-2xl border border-border bg-card/50">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  const chartData = data.map(d => ({
    ...d,
    difference: d.entry_cumulative - d.exit_cumulative
  }))

  return (
    <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
      <h3 className="mb-6 text-lg font-semibold text-foreground">Campus Vehicle Count (Entry vs Exit)</h3>
      <div className="h-[350px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
            <XAxis dataKey="time" tick={{ fill: "#A1A1AA", fontSize: 12 }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fill: "#A1A1AA", fontSize: 12 }} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{ borderRadius: '12px', border: '1px solid hsl(var(--border))', backgroundColor: 'hsl(var(--card))', color: 'hsl(var(--foreground))' }}
              formatter={(value: number, name: string, props: any) => {
                if (name === 'difference') return [value, 'Current Vehicles on Campus'];
                return [value, name === 'entry_cumulative' ? 'Total Entries' : 'Total Exits'];
              }}
            />
            <Legend />
            <Line type="linear" dataKey="entry_cumulative" name="Cumulative Entry" stroke="#3b82f6" strokeWidth={2} dot={true} />
            <Line type="linear" dataKey="exit_cumulative" name="Cumulative Exit" stroke="#ef4444" strokeWidth={2} dot={true} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
