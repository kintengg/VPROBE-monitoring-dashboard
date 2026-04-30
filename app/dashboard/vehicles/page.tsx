"use client"

import { useCallback, useEffect, useState } from "react"
import { useUploadQueue } from "@/components/uploads/upload-queue-provider"
import { Button } from "@/components/ui/button"
import { FootageDatePicker } from "@/components/ui/footage-date-picker"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { AlertCircle, Clock, Download, FileCode, Loader2, RefreshCw, Settings2, Upload, BarChart3, LineChart as LineChartIcon } from "lucide-react"
import {
  downloadDashboardReport,
  getCurrentModel,
  getDashboardTraffic,
  getLocations,
  uploadModel,
  type ModelInfo,
} from "@/lib/api"
import { VehicleKPICards, VehicleCountChart, VehicleLOSChart, VehicleAllGatesChart, VehicleCampusCountChart } from "@/components/dashboard/vehicle-dashboard-components"

export default function DashboardPage() {
  const { settledUploadsVersion } = useUploadQueue()
  const [selectedDate, setSelectedDate] = useState("2026-03-15")
  const [timeRange, setTimeRange] = useState("whole-day")
  const [startTime, setStartTime] = useState<string>("00:00")
  const [modelDialogOpen, setModelDialogOpen] = useState(false)
  const [modelFile, setModelFile] = useState<File | null>(null)
  
  const [traffic, setTraffic] = useState<any>(null)
  const [footageDates, setFootageDates] = useState<string[]>([])
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  
  const [dashboardLoading, setDashboardLoading] = useState(true)
  const [modelLoading, setModelLoading] = useState(true)
  const [modelUploading, setModelUploading] = useState(false)
  const [reportExporting, setReportExporting] = useState(false)
  const [dashboardError, setDashboardError] = useState<string | null>(null)
  const [modelError, setModelError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  const [countChartType, setCountChartType] = useState<'bar' | 'line'>('bar')
  const [losChartType, setLosChartType] = useState<'bar' | 'line'>('line')

  const loadDashboard = useCallback(async () => {
    setDashboardLoading(true)
    try {
      const trafficResponse = await getDashboardTraffic(selectedDate || undefined, timeRange, undefined, 0, undefined, startTime, 'vehicle')
      setTraffic(trafficResponse)
      setDashboardError(null)
    } catch (error) {
      setTraffic(null)
      setDashboardError(error instanceof Error ? error.message : "Failed to load dashboard data.")
    } finally {
      setDashboardLoading(false)
    }
  }, [selectedDate, timeRange, startTime])

  const loadModelInfo = useCallback(async () => {
    setModelLoading(true)
    try {
      const response = await getCurrentModel()
      setModelInfo(response)
      setModelError(null)
    } catch (error) {
      setModelError(error instanceof Error ? error.message : "Failed to load model information.")
    } finally {
      setModelLoading(false)
    }
  }, [])

  const loadFootageDates = useCallback(async () => {
    try {
      const response = await getLocations(undefined, 'vehicle')
      setFootageDates(Array.from(new Set(response.flatMap((location) => location.videos.map((video) => video.date)))).sort())
    } catch {
      // Ignore
    }
  }, [])

  useEffect(() => {
    void loadDashboard()
  }, [loadDashboard])

  useEffect(() => {
    void loadModelInfo()
  }, [loadModelInfo])

  useEffect(() => {
    void loadFootageDates()
  }, [loadFootageDates])

  useEffect(() => {
    if (settledUploadsVersion === 0) return
    void loadDashboard()
    void loadFootageDates()
  }, [loadDashboard, loadFootageDates, settledUploadsVersion])

  const handleModelUpload = async () => {
    if (!modelFile) return
    setModelUploading(true)
    try {
      const response = await uploadModel(modelFile)
      setModelInfo(response)
      setModelError(null)
      setModelDialogOpen(false)
      setModelFile(null)
    } catch (error) {
      setModelError(error instanceof Error ? error.message : "Failed to upload model.")
    } finally {
      setModelUploading(false)
    }
  }

  const handleExportReport = async () => {
    setReportExporting(true)
    setActionError(null)
    try {
      const { blob, filename } = await downloadDashboardReport(selectedDate, timeRange)
      const downloadUrl = window.URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = downloadUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(downloadUrl)
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Failed to export report.")
    } finally {
      setReportExporting(false)
    }
  }

  const currentModelLabel = modelInfo?.currentModel ?? "No model uploaded yet"
  const currentModelTimestamp = modelInfo?.uploadedAt ? new Date(modelInfo.uploadedAt).toLocaleString("en-US") : null
  const bannerError = dashboardError ?? modelError ?? actionError

  // Generate 24 hours options for start time dropdown
  const hourOptions = Array.from({ length: 24 }).map((_, i) => {
    const hour = i.toString().padStart(2, '0')
    const displayHour = i === 0 ? '12 AM' : i < 12 ? `${i} AM` : i === 12 ? '12 PM' : `${i - 12} PM`
    return { value: `${hour}:00`, label: displayHour }
  })

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-border bg-card/50 px-6 py-4 backdrop-blur-sm">
        <div>
          <h1 className="text-xl font-semibold text-foreground">System Vehicle Analytics Dashboard</h1>
          <p className="text-sm text-muted-foreground">Real-time vehicle tracking metrics</p>
        </div>

        <div className="flex items-center gap-3">
          <FootageDatePicker
            value={selectedDate}
            onChange={(v) => setSelectedDate(v)}
            highlightedDates={footageDates}
            placeholder="Select date"
          />

          <Select value={timeRange} onValueChange={(v) => setTimeRange(v)}>
            <SelectTrigger className="w-40 rounded-2xl border-border bg-secondary text-foreground">
              <Clock className="mr-2 h-4 w-4 text-muted-foreground" />
              <SelectValue placeholder="Select range" />
            </SelectTrigger>
            <SelectContent className="rounded-xl border-border bg-popover">
              <SelectItem value="whole-day" className="rounded-lg text-foreground">Whole Day</SelectItem>
              <SelectItem value="12h" className="rounded-lg text-foreground">12 Hours</SelectItem>
              <SelectItem value="6h" className="rounded-lg text-foreground">6 Hours</SelectItem>
              <SelectItem value="1h" className="rounded-lg text-foreground">1 Hour</SelectItem>
            </SelectContent>
          </Select>

          {timeRange !== "whole-day" && (
            <Select value={startTime} onValueChange={setStartTime}>
              <SelectTrigger className="w-36 rounded-2xl border-border bg-secondary text-foreground">
                <SelectValue placeholder="Start time" />
              </SelectTrigger>
              <SelectContent className="h-64 rounded-xl border-border bg-popover">
                {hourOptions.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value} className="rounded-lg text-foreground">
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}

          <Button
            variant="outline"
            className="rounded-2xl border-border px-4 text-foreground hover:bg-secondary"
            onClick={() => {
              void loadDashboard()
            }}
          >
            <RefreshCw className={`mr-2 h-4 w-4 ${dashboardLoading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </header>

      <div className="flex-1 space-y-6 overflow-auto p-6">
        {bannerError && (
          <div className="flex items-start gap-3 rounded-2xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{bannerError}</span>
          </div>
        )}

        <VehicleKPICards summary={traffic?.summary} loading={dashboardLoading} />

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          <div className="space-y-2">
            <div className="flex justify-end gap-2">
              <Button variant={countChartType === 'bar' ? 'default' : 'outline'} size="sm" onClick={() => setCountChartType('bar')} className="h-8">
                <BarChart3 className="mr-2 h-4 w-4" /> Bar
              </Button>
              <Button variant={countChartType === 'line' ? 'default' : 'outline'} size="sm" onClick={() => setCountChartType('line')} className="h-8">
                <LineChartIcon className="mr-2 h-4 w-4" /> Line
              </Button>
            </div>
            <VehicleCountChart data={traffic?.series ?? []} loading={dashboardLoading} type={countChartType} />
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-end gap-2">
              <Button variant={losChartType === 'bar' ? 'default' : 'outline'} size="sm" onClick={() => setLosChartType('bar')} className="h-8">
                <BarChart3 className="mr-2 h-4 w-4" /> Bar
              </Button>
              <Button variant={losChartType === 'line' ? 'default' : 'outline'} size="sm" onClick={() => setLosChartType('line')} className="h-8">
                <LineChartIcon className="mr-2 h-4 w-4" /> Line
              </Button>
            </div>
            <VehicleLOSChart data={traffic?.series ?? []} loading={dashboardLoading} type={losChartType} />
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-end gap-2 opacity-0">
              <Button variant="outline" size="sm" className="h-8 pointer-events-none">Spacer</Button>
            </div>
            <VehicleAllGatesChart data={traffic?.series ?? []} loading={dashboardLoading} />
          </div>

          <div className="space-y-2">
            <div className="flex justify-end gap-2 opacity-0">
              <Button variant="outline" size="sm" className="h-8 pointer-events-none">Spacer</Button>
            </div>
            <VehicleCampusCountChart data={traffic?.series ?? []} loading={dashboardLoading} />
          </div>
        </div>

      </div>
    </div>
  )
}
