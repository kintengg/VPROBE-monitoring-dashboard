"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { KPICards } from "@/components/dashboard/kpi-cards"
import { PedestrianChart } from "@/components/dashboard/pedestrian-chart"
import { OcclusionTrendsChart } from "@/components/dashboard/occlusion-trends-chart"
import { OcclusionMap } from "@/components/dashboard/occlusion-map"
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
import { AlertCircle, Clock, Download, FileCode, Loader2, RefreshCw, Settings2, Upload } from "lucide-react"
import {
  downloadDashboardReport,
  getAISynthesis,
  getCurrentModel,
  getDashboardLOS,
  getDashboardOcclusion,
  getDashboardOcclusionTrends,
  getDashboardSummary,
  getDashboardTraffic,
  getDashboardTrafficByLocation,
  getInferConfigChoices,
  getInferenceStatus,
  getLocations,
  updateModelSettings,
  uploadInferenceRequirement,
  uploadModel,
  type AISynthesisResponse,
  type DashboardSummary,
  type InferConfigList,
  type InferenceRequirementType,
  type InferenceStatus,
  type LocationRecord,
  type ModelInfo,
  type PTSIMapResponse,
  type PTSITrendResponse,
  type TrafficByLocationResponse,
  type TrafficResponse,
} from "@/lib/api"

const TIME_RANGE_OPTIONS = [
  { value: "whole-day", label: "Whole day (24h)" },
  { value: "12h", label: "12 hours" },
  { value: "6h", label: "6 hours" },
  { value: "4h", label: "4 hours" },
  { value: "3h", label: "3 hours" },
  { value: "2h", label: "2 hours" },
  { value: "1h", label: "1 hour" },
  { value: "30m", label: "30 minutes" },
] as const

const START_TIME_OPTIONS = Array.from({ length: 48 }, (_, index) => {
  const totalMinutes = index * 30
  const hour24 = Math.floor(totalMinutes / 60)
  const hours = String(hour24).padStart(2, "0")
  const minutes = String(totalMinutes % 60).padStart(2, "0")
  const value = `${hours}:${minutes}`
  const suffix = hour24 >= 12 ? "PM" : "AM"
  const hour12 = hour24 % 12 || 12
  const label = `${hour12}:${minutes} ${suffix}`
  return { value, label }
})

const LOS_BY_NUMERIC_RANK: Record<number, string> = {
  1: "A",
  2: "B",
  3: "C",
  4: "D",
  5: "E",
  6: "F",
}

const deriveLosGradeFromSeries = (series?: TrafficResponse["series"]): string | null => {
  const losValues = (series ?? [])
    .map((point) => point.los)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value))

  if (losValues.length === 0) {
    return null
  }

  const averageRank = Math.round(losValues.reduce((sum, value) => sum + value, 0) / losValues.length)
  return LOS_BY_NUMERIC_RANK[averageRank] ?? null
}

const getCurrentLocalDate = () => {
  const now = new Date()
  const timezoneOffsetMilliseconds = now.getTimezoneOffset() * 60 * 1000
  return new Date(now.getTime() - timezoneOffsetMilliseconds).toISOString().slice(0, 10)
}

const hasFootageForDate = (location: LocationRecord, date: string) =>
  location.videos.some((video) => video.date === date)

const hasAnyFootage = (location: LocationRecord) => location.videos.length > 0

const pickPreferredLocationId = (locations: LocationRecord[], date: string): string => {
  const withSelectedDateFootage = locations.find((location) => hasFootageForDate(location, date))
  if (withSelectedDateFootage) {
    return withSelectedDateFootage.id
  }

  const withAnyFootage = locations.find((location) => hasAnyFootage(location))
  if (withAnyFootage) {
    return withAnyFootage.id
  }

  return locations[0]?.id ?? ""
}

export default function VehicleDashboardPage() {
  const { settledUploadsVersion } = useUploadQueue()
  const [selectedDate, setSelectedDate] = useState("")
  const [timeRange, setTimeRange] = useState("whole-day")
  const [startTime, setStartTime] = useState("06:00")
  const [focusTime, setFocusTime] = useState<string | undefined>(undefined)
  const [zoomLevel, setZoomLevel] = useState(0)
  const [vehicleChartType, setVehicleChartType] = useState<"line" | "bar">("line")
  const [losChartType, setLosChartType] = useState<"line" | "bar">("bar")
  const [allGatesVehicleChartType, setAllGatesVehicleChartType] = useState<"line" | "bar">("line")
  const [inOutChartType, setInOutChartType] = useState<"line" | "bar">("line")
  const [modelDialogOpen, setModelDialogOpen] = useState(false)
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [selectedInferConfig, setSelectedInferConfig] = useState("")
  const [modelBatchSize, setModelBatchSize] = useState("16")
  const [inferConfigChoices, setInferConfigChoices] = useState<InferConfigList>({ options: [], defaultConfig: null })
  const [requirementFile, setRequirementFile] = useState<File | null>(null)
  const [requirementType, setRequirementType] = useState<InferenceRequirementType>("infer-config")
  const [summary, setSummary] = useState<DashboardSummary | null>(null)
  const [traffic, setTraffic] = useState<TrafficResponse | null>(null)
  const [trafficByLocation, setTrafficByLocation] = useState<TrafficByLocationResponse | null>(null)
  const [losTraffic, setLosTraffic] = useState<TrafficResponse | null>(null)
  const [occlusionTrends, setOcclusionTrends] = useState<PTSITrendResponse | null>(null)
  const [occlusion, setOcclusion] = useState<PTSIMapResponse | null>(null)
  const [hourFilter, setHourFilter] = useState("all")
  const [synthesis, setSynthesis] = useState<AISynthesisResponse | null>(null)
  const [locations, setLocations] = useState<LocationRecord[]>([])
  const [selectedLocationId, setSelectedLocationId] = useState<string>("")
  const [userSelectedDate, setUserSelectedDate] = useState(false)
  const [userSelectedLocation, setUserSelectedLocation] = useState(false)
  const [footageDates, setFootageDates] = useState<string[]>([])
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [inferenceStatus, setInferenceStatus] = useState<InferenceStatus | null>(null)
  const [dashboardLoading, setDashboardLoading] = useState(true)
  const [modelLoading, setModelLoading] = useState(true)
  const [modelUploading, setModelUploading] = useState(false)
  const [modelSettingsSaving, setModelSettingsSaving] = useState(false)
  const [requirementUploading, setRequirementUploading] = useState(false)
  const [reportExporting, setReportExporting] = useState(false)
  const [dashboardError, setDashboardError] = useState<string | null>(null)
  const [modelError, setModelError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [requirementUploadMessage, setRequirementUploadMessage] = useState<string | null>(null)

  const loadDashboard = useCallback(async () => {
    if (!selectedDate) {
      return
    }

    setDashboardLoading(true)
    try {
      const [summaryResponse, trafficResponse, trafficByLocationResponse, losTrafficResponse, occlusionTrendsResponse, occlusionResponse, synthesisResponse] = await Promise.all([
        getDashboardSummary(selectedDate),
        getDashboardTraffic(selectedDate, timeRange, focusTime, zoomLevel, startTime, selectedLocationId || undefined),
        getDashboardTrafficByLocation(selectedDate, timeRange, focusTime, zoomLevel, startTime),
        getDashboardLOS(selectedDate, timeRange, focusTime, zoomLevel, selectedLocationId || undefined, startTime),
        getDashboardOcclusionTrends(selectedDate, timeRange, focusTime, zoomLevel, startTime),
        getDashboardOcclusion(selectedDate, timeRange, startTime),
        getAISynthesis(selectedDate, timeRange, startTime),
      ])

      setSummary(summaryResponse)
      setTraffic(trafficResponse)
      setTrafficByLocation(trafficByLocationResponse)
      setLosTraffic(losTrafficResponse)
      setOcclusionTrends(occlusionTrendsResponse)
      setOcclusion(occlusionResponse)
      setSynthesis(synthesisResponse)
      setDashboardError(null)
    } catch (error) {
      setSummary(null)
      setTraffic(null)
      setTrafficByLocation(null)
      setLosTraffic(null)
      setOcclusionTrends(null)
      setOcclusion(null)
      setSynthesis(null)
      setDashboardError(error instanceof Error ? error.message : "Failed to load dashboard data.")
    } finally {
      setDashboardLoading(false)
    }
  }, [focusTime, selectedDate, selectedLocationId, startTime, timeRange, zoomLevel])

  const loadModelInfo = useCallback(async () => {
    setModelLoading(true)
    try {
      const [modelResponse, inferenceResponse, inferConfigResponse] = await Promise.all([
        getCurrentModel(),
        getInferenceStatus(),
        getInferConfigChoices(),
      ])
      setModelInfo(modelResponse)
      setInferenceStatus(inferenceResponse)
      setInferConfigChoices(inferConfigResponse)
      const configFromModel = modelResponse.inferConfig ?? ""
      const nextSelectedConfig =
        (configFromModel && inferConfigResponse.options.includes(configFromModel) ? configFromModel : "") ||
        (inferConfigResponse.defaultConfig ?? "")
      setSelectedInferConfig(nextSelectedConfig)
      setModelBatchSize(String(modelResponse.batchSize ?? 16))
      setModelError(null)
    } catch (error) {
      setModelError(error instanceof Error ? error.message : "Failed to load model information.")
    } finally {
      setModelLoading(false)
    }
  }, [])

  const loadFootageDates = useCallback(async () => {
    try {
      const response = await getLocations()
      setLocations(response)

      const allDates = Array.from(new Set(response.flatMap((location) => location.videos.map((video) => video.date)))).sort()
      const latestFootageDate = allDates.at(-1)
      const fallbackDate = getCurrentLocalDate()
      const preferredDate = latestFootageDate ?? fallbackDate

      setFootageDates(allDates)

      if (!userSelectedDate || !selectedDate) {
        setSelectedDate(preferredDate)
      }

      if (response.length === 0) {
        setSelectedLocationId("")
        return
      }

      const effectiveDate = userSelectedDate && selectedDate ? selectedDate : preferredDate
      const preferredLocationId = pickPreferredLocationId(response, effectiveDate)
      const currentSelection = response.find((location) => location.id === selectedLocationId)

      if (!currentSelection) {
        setSelectedLocationId(preferredLocationId)
      } else if (!userSelectedLocation && !hasFootageForDate(currentSelection, effectiveDate)) {
        setSelectedLocationId(preferredLocationId)
      }
    } catch {
      // Leave existing date highlights untouched if this auxiliary request fails.
    }
  }, [selectedDate, selectedLocationId, userSelectedDate, userSelectedLocation])

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
    if (settledUploadsVersion === 0) {
      return
    }

    void loadDashboard()
    void loadFootageDates()
  }, [loadDashboard, loadFootageDates, settledUploadsVersion])

  useEffect(() => {
    if (hourFilter !== "all" && !occlusion?.availableHours.includes(hourFilter)) {
      setHourFilter("all")
    }
  }, [hourFilter, occlusion])

  const handleModelUpload = async () => {
    const parsedBatchSize = Number.parseInt(modelBatchSize, 10)
    if (!modelFile || !selectedInferConfig || !Number.isFinite(parsedBatchSize) || parsedBatchSize < 1 || parsedBatchSize > 256) {
      return
    }

    setModelUploading(true)
    try {
      await uploadModel(modelFile, selectedInferConfig, parsedBatchSize)
      await loadModelInfo()
      setModelError(null)
      setModelDialogOpen(false)
      setModelFile(null)
    } catch (error) {
      setModelError(error instanceof Error ? error.message : "Failed to upload model.")
    } finally {
      setModelUploading(false)
    }
  }

  const handleModelSettingsSave = async () => {
    const parsedBatchSize = Number.parseInt(modelBatchSize, 10)
    if (!selectedInferConfig || !Number.isFinite(parsedBatchSize) || parsedBatchSize < 1 || parsedBatchSize > 256) {
      setModelError("Batch size must be a number between 1 and 256.")
      return
    }

    setModelSettingsSaving(true)
    try {
      await updateModelSettings({
        inferConfig: selectedInferConfig,
        batchSize: parsedBatchSize,
      })
      await loadModelInfo()
      setModelError(null)
    } catch (error) {
      setModelError(error instanceof Error ? error.message : "Failed to save model settings.")
    } finally {
      setModelSettingsSaving(false)
    }
  }

  const handleRequirementUpload = async () => {
    if (!requirementFile) {
      return
    }

    setRequirementUploading(true)
    try {
      const response = await uploadInferenceRequirement({
        file: requirementFile,
        requirementType,
      })
      setRequirementUploadMessage(`${response.message} Saved to ${response.savedPath}`)
      setActionError(null)
      setRequirementFile(null)
      await loadModelInfo()
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "Failed to upload requirement file.")
    } finally {
      setRequirementUploading(false)
    }
  }

  const handleExportReport = async () => {
    setReportExporting(true)
    setActionError(null)

    try {
      const { blob, filename } = await downloadDashboardReport(selectedDate, timeRange, startTime)
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
  const currentInferConfigLabel = modelInfo?.inferConfig ?? inferConfigChoices.defaultConfig ?? "Not selected"
  const currentBatchSizeLabel = typeof modelInfo?.batchSize === "number" ? String(modelInfo.batchSize) : "16"
  const currentModelTimestamp = modelInfo?.uploadedAt ? new Date(modelInfo.uploadedAt).toLocaleString("en-US") : null
  const inferenceReady = Boolean(inferenceStatus?.ready)
  const missingRequiredPath = inferenceStatus?.missingFixedPath
  const bannerError = dashboardError ?? modelError ?? actionError

  const selectedDateClipCount = useMemo(
    () => locations.reduce((count, location) => count + location.videos.filter((video) => video.date === selectedDate).length, 0),
    [locations, selectedDate],
  )

  const averageLosLabel = useMemo(() => {
    return deriveLosGradeFromSeries(losTraffic?.series) ?? "--"
  }, [losTraffic])

  const selectedLocationName = useMemo(
    () => locations.find((location) => location.id === selectedLocationId)?.name ?? "Selected gate",
    [locations, selectedLocationId],
  )

  const handleDateChange = (value: string) => {
    setUserSelectedDate(true)
    setSelectedDate(value)
    setUserSelectedLocation(false)
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  const handleTimeRangeChange = (value: string) => {
    setTimeRange(value)
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  const handleStartTimeChange = (value: string) => {
    setStartTime(value)
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  const handleLocationChange = (value: string) => {
    setUserSelectedLocation(true)
    setSelectedLocationId(value)
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  const handleAnalyticsZoom = (time: string) => {
    const canZoomInAnyChart = Boolean(traffic?.canZoomIn) || Boolean(losTraffic?.canZoomIn) || Boolean(occlusionTrends?.canZoomIn)
    if (!canZoomInAnyChart) {
      return
    }

    setFocusTime(time)
    setZoomLevel((current) => current + 1)
  }

  const handleResetZoom = () => {
    setFocusTime(undefined)
    setZoomLevel(0)
  }

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-border bg-card/50 px-4 py-4 backdrop-blur-sm sm:px-6">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-4">
          <div className="shrink-0">
            <h1 className="text-xl font-semibold text-foreground">System Analytics: Vehicles</h1>
            <p className="text-sm text-muted-foreground">Real-time vehicle movement and LOS metrics</p>
          </div>

          <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_auto] xl:items-center xl:gap-4">
            <div className="grid w-full grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-[minmax(0,15rem)_12rem_11rem_14rem] xl:gap-2">
              <div className="w-full min-w-0">
                <FootageDatePicker
                  value={selectedDate}
                  onChange={handleDateChange}
                  highlightedDates={footageDates}
                  placeholder="Select date"
                  className="!h-11 min-h-11"
                />
              </div>

              <Select value={timeRange} onValueChange={handleTimeRangeChange}>
                <SelectTrigger className="!h-11 min-h-11 w-full rounded-2xl border-border bg-secondary py-0 text-foreground">
                  <Clock className="mr-2 h-4 w-4 text-muted-foreground" />
                  <SelectValue placeholder="Select time range" />
                </SelectTrigger>
                <SelectContent className="rounded-xl border-border bg-popover">
                  {TIME_RANGE_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value} className="rounded-lg text-foreground">
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select value={startTime} onValueChange={handleStartTimeChange}>
                <SelectTrigger className="!h-11 min-h-11 w-full rounded-2xl border-border bg-secondary py-0 text-foreground">
                  <SelectValue placeholder="Start time" />
                </SelectTrigger>
                <SelectContent className="max-h-80 rounded-xl border-border bg-popover">
                  {START_TIME_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value} className="rounded-lg text-foreground">
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select value={selectedLocationId} onValueChange={handleLocationChange}>
                <SelectTrigger className="!h-11 min-h-11 w-full rounded-2xl border-border bg-secondary py-0 text-foreground">
                  <SelectValue placeholder="Select location" />
                </SelectTrigger>
                <SelectContent className="rounded-xl border-border bg-popover">
                  {locations.map((location) => (
                    <SelectItem key={location.id} value={location.id} className="rounded-lg text-foreground">
                      {location.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="flex w-full flex-col gap-3 sm:flex-row sm:flex-wrap sm:justify-end xl:w-auto xl:flex-nowrap xl:justify-end">
              <Dialog open={modelDialogOpen} onOpenChange={setModelDialogOpen}>
                <DialogTrigger asChild>
                  <Button variant="outline" className="!h-11 min-h-11 w-full rounded-2xl border-border px-4 py-0 text-foreground hover:bg-secondary sm:w-auto sm:min-w-[9.25rem]">
                    <Settings2 className="mr-2 h-4 w-4" />
                    Edit Model
                  </Button>
                </DialogTrigger>
                <DialogContent className="max-w-md rounded-3xl border-border bg-card">
                  <DialogHeader>
                    <DialogTitle className="text-foreground">Detection Model Settings</DialogTitle>
                    <DialogDescription className="text-muted-foreground">
                      Upload a PyTorch model file (.pt or .pth) for vehicle detection
                    </DialogDescription>
                  </DialogHeader>

                  <div className="space-y-3 pt-3">
                    <div className="rounded-2xl border border-border bg-secondary p-3">
                      <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/20">
                          <FileCode className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-foreground">Current Model</p>
                          <p className="text-xs text-muted-foreground">{modelLoading ? "Loading model..." : currentModelLabel}</p>
                          <p className="mt-1 text-[11px] text-muted-foreground">Config: {currentInferConfigLabel}</p>
                          <p className="mt-1 text-[11px] text-muted-foreground">Batch size: {currentBatchSizeLabel}</p>
                          {currentModelTimestamp && (
                            <p className="mt-1 text-[11px] text-muted-foreground">Uploaded {currentModelTimestamp}</p>
                          )}
                          {!modelLoading && (
                            <p className={`mt-2 text-[11px] font-medium ${inferenceReady ? "text-emerald-600" : "text-amber-600"}`}>
                              {inferenceReady ? "Inference ready for video processing" : "Inference not ready yet"}
                            </p>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="rounded-2xl border border-border bg-secondary p-3">
                      <p className="text-sm font-medium text-foreground">Inference Requirements</p>
                      <p className="mt-1 text-xs text-muted-foreground">Required before processing starts.</p>

                      <ul className="mt-2 space-y-1 text-xs text-foreground">
                        <li>Model uploaded (.pt/.pth): {inferenceStatus?.modelExists ? "OK" : "Missing"}</li>
                        <li>RT-DETR pipeline files found: {inferenceStatus?.installed ? "OK" : "Missing"}</li>
                        <li>Ready to process videos: {inferenceReady ? "Yes" : "No"}</li>
                      </ul>

                      <p className="mt-2 text-[11px] text-muted-foreground">Needs: infer config, counting config, and model weights.</p>

                      {missingRequiredPath && (
                        <p className="mt-3 text-[11px] text-amber-700">
                          Missing required path: {missingRequiredPath}
                        </p>
                      )}

                      <div className="mt-3 space-y-2 rounded-xl border border-border/70 bg-background/60 p-3">
                        <p className="text-xs font-medium text-foreground">Upload Missing Requirement File</p>

                        <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_140px] sm:items-end">
                          <Select value={requirementType} onValueChange={(value) => setRequirementType(value as InferenceRequirementType)}>
                            <SelectTrigger className="h-9 rounded-xl border-border bg-background text-xs text-foreground">
                              <SelectValue placeholder="Select requirement type" />
                            </SelectTrigger>
                            <SelectContent className="rounded-xl border-border bg-popover">
                              <SelectItem value="infer-config" className="rounded-lg text-foreground">Infer Config (.yml/.yaml)</SelectItem>
                              <SelectItem value="annotations" className="rounded-lg text-foreground">Annotations (.json)</SelectItem>
                            </SelectContent>
                          </Select>

                          <div className="space-y-2">
                            <input
                              type="file"
                              accept={requirementType === "infer-config" ? ".yml,.yaml" : ".json"}
                              onChange={(e) => setRequirementFile(e.target.files?.[0] || null)}
                              className="hidden"
                              id="requirement-upload"
                            />
                            <label
                              htmlFor="requirement-upload"
                              className="flex h-9 cursor-pointer items-center justify-center rounded-xl border border-border bg-background px-3 text-xs font-medium text-foreground transition-colors hover:bg-secondary"
                            >
                              Choose File
                            </label>
                          </div>
                        </div>

                        <p className="text-[11px] text-muted-foreground">{requirementFile ? `Selected: ${requirementFile.name}` : "No requirement file selected."}</p>

                        <Button
                          onClick={handleRequirementUpload}
                          disabled={!requirementFile || requirementUploading}
                          variant="outline"
                          className="h-9 w-full rounded-xl border-border text-xs text-foreground hover:bg-secondary"
                        >
                          {requirementUploading ? (
                            <>
                              <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                              Uploading Requirement...
                            </>
                          ) : (
                            "Upload Requirement File"
                          )}
                        </Button>

                        {requirementUploadMessage && <p className="text-[11px] text-emerald-700">{requirementUploadMessage}</p>}
                      </div>
                    </div>

                    <div className="space-y-1.5">
                      <label className="text-sm font-medium text-foreground">Upload New Model</label>

                      <div className="grid grid-cols-1 gap-2 sm:grid-cols-[minmax(0,1fr)_140px] sm:items-start">
                        <div className="space-y-2">
                          <label className="text-xs font-medium text-muted-foreground">Inference Config (.yml)</label>
                          <Select value={selectedInferConfig} onValueChange={setSelectedInferConfig}>
                            <SelectTrigger className="h-9 rounded-xl border-border bg-background text-xs text-foreground">
                              <SelectValue placeholder="Select infer config" />
                            </SelectTrigger>
                            <SelectContent className="rounded-xl border-border bg-popover">
                              {inferConfigChoices.options.map((configName) => (
                                <SelectItem key={configName} value={configName} className="rounded-lg text-foreground">
                                  {configName}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <label className="text-xs font-medium text-muted-foreground">Batch Size</label>
                          <input
                            type="number"
                            min={1}
                            max={256}
                            value={modelBatchSize}
                            onChange={(event) => {
                              setModelBatchSize(event.target.value)
                            }}
                            className="h-9 rounded-xl border border-border bg-background px-3 text-xs text-foreground"
                          />
                        </div>
                      </div>
                      <p className="text-[11px] text-muted-foreground">Batch Size controls how many frames run together. Higher values use more VRAM.</p>
                      <Button
                        onClick={handleModelSettingsSave}
                        disabled={modelSettingsSaving || !selectedInferConfig || !modelBatchSize}
                        variant="outline"
                        className="h-9 w-full rounded-xl border-border text-xs text-foreground hover:bg-secondary"
                      >
                        {modelSettingsSaving ? (
                          <>
                            <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                            Saving Settings...
                          </>
                        ) : (
                          "Save Settings"
                        )}
                      </Button>

                      <div
                        className={`rounded-2xl border-2 border-dashed p-4 text-center transition-colors ${
                          modelFile ? "border-accent bg-accent/10" : "border-border hover:border-muted-foreground"
                        }`}
                      >
                        <input
                          type="file"
                          accept=".pt,.pth"
                          onChange={(e) => setModelFile(e.target.files?.[0] || null)}
                          className="hidden"
                          id="model-upload"
                        />
                        <label htmlFor="model-upload" className="cursor-pointer">
                          <Upload className={`mx-auto mb-1.5 h-6 w-6 ${modelFile ? "text-accent" : "text-muted-foreground"}`} />
                          {modelFile ? (
                            <p className="text-sm font-medium text-accent">{modelFile.name}</p>
                          ) : (
                            <>
                              <p className="text-sm text-foreground">Click to upload .pt or .pth file</p>
                              <p className="mt-1 text-xs text-muted-foreground">PyTorch model weights</p>
                            </>
                          )}
                        </label>
                      </div>
                    </div>

                    <Button
                      onClick={handleModelUpload}
                      disabled={!modelFile || !selectedInferConfig || !modelBatchSize || modelUploading}
                      className="w-full rounded-2xl bg-primary text-primary-foreground hover:bg-primary/90"
                    >
                      {modelUploading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Uploading Model...
                        </>
                      ) : (
                        "Update Model"
                      )}
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>

              <Button
                variant="outline"
                className="!h-11 min-h-11 w-full rounded-2xl border-border px-4 py-0 text-foreground hover:bg-secondary sm:w-auto sm:min-w-[9.25rem]"
                onClick={() => {
                  void loadDashboard()
                  void loadModelInfo()
                  void loadFootageDates()
                }}
              >
                <RefreshCw className={`mr-2 h-4 w-4 ${dashboardLoading || modelLoading ? "animate-spin" : ""}`} />
                Refresh
              </Button>

              <Button
                className="!h-11 min-h-11 w-full rounded-2xl bg-accent px-4 py-0 text-accent-foreground shadow-elevated-sm hover:bg-accent/90 sm:w-auto sm:min-w-[10.5rem]"
                onClick={() => {
                  void handleExportReport()
                }}
                disabled={reportExporting}
              >
                {reportExporting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Download className="mr-2 h-4 w-4" />}
                {reportExporting ? "Exporting..." : "Export Report"}
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex-1 space-y-6 overflow-auto p-6">
        {bannerError && (
          <div className="flex items-start gap-3 rounded-2xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{bannerError}</span>
          </div>
        )}

        <KPICards
          summary={summary}
          loading={dashboardLoading}
        />

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[65%_35%]">
          <div className="space-y-6">
            <PedestrianChart
              title={selectedLocationId ? `Vehicle Count – ${selectedLocationName}` : "Vehicle Count"}
              description={
                selectedLocationId
                  ? `Cumulative vehicle count for ${selectedLocationName} over the selected time window.`
                  : "Select a gate to view its vehicle count trend."
              }
              timeRange={timeRange}
              selectedDate={selectedDate}
              data={traffic?.series ?? []}
              metricKey="cumulativeUniquePedestrians"
              metricLabel="Vehicle Count"
              seriesColor="#22C55E"
              bucketMinutes={traffic?.bucketMinutes ?? 60}
              zoomLevel={traffic?.zoomLevel ?? 0}
              canZoomIn={traffic?.canZoomIn ?? false}
              focusTime={traffic?.focusTime}
              windowStart={traffic?.windowStart}
              windowEnd={traffic?.windowEnd}
              loading={dashboardLoading}
              onTimeSelect={handleAnalyticsZoom}
              onResetZoom={handleResetZoom}
              chartType={vehicleChartType}
              onChartTypeChange={setVehicleChartType}
            />
            <PedestrianChart
              title={selectedLocationId ? `LOS – ${selectedLocationName}` : "LOS"}
              description={
                selectedLocationId
                  ? `Level of Service trend for ${selectedLocationName} across the selected time window.`
                  : "Select a gate from the filter above to view its Level of Service trend."
              }
              timeRange={timeRange}
              selectedDate={selectedDate}
              data={losTraffic?.series ?? []}
              metricKey="los"
              metricLabel="LOS"
              seriesColor="#06B6D4"
              bucketMinutes={losTraffic?.bucketMinutes ?? 60}
              zoomLevel={losTraffic?.zoomLevel ?? 0}
              canZoomIn={losTraffic?.canZoomIn ?? false}
              focusTime={losTraffic?.focusTime}
              windowStart={losTraffic?.windowStart}
              windowEnd={losTraffic?.windowEnd}
              loading={dashboardLoading}
              onTimeSelect={handleAnalyticsZoom}
              onResetZoom={handleResetZoom}
              chartType={losChartType}
              onChartTypeChange={setLosChartType}
            />
            <PedestrianChart
              title="Vehicle Count (All Gates)"
              description="Gate-by-gate cumulative vehicle count for the selected date and time window."
              timeRange={timeRange}
              selectedDate={selectedDate}
              data={trafficByLocation?.series ?? []}
              metricKey="cumulativeUniquePedestrians"
              metricLabel="Vehicle Count"
              seriesColor="#22C55E"
              bucketMinutes={trafficByLocation?.bucketMinutes ?? 60}
              zoomLevel={trafficByLocation?.zoomLevel ?? 0}
              canZoomIn={trafficByLocation?.canZoomIn ?? false}
              focusTime={trafficByLocation?.focusTime}
              windowStart={trafficByLocation?.windowStart}
              windowEnd={trafficByLocation?.windowEnd}
              loading={dashboardLoading}
              onTimeSelect={handleAnalyticsZoom}
              onResetZoom={handleResetZoom}
              chartType={allGatesVehicleChartType}
              onChartTypeChange={setAllGatesVehicleChartType}
              legendPosition="top"
              useLosLineColors={false}
            />
            <PedestrianChart
              title="LOS (All Gates)"
              description="Gate-by-gate Level of Service trend for the selected date and time window."
              timeRange={timeRange}
              selectedDate={selectedDate}
              data={trafficByLocation?.series ?? []}
              metricKey="los"
              metricLabel="LOS"
              seriesColor="#06B6D4"
              bucketMinutes={trafficByLocation?.bucketMinutes ?? 60}
              zoomLevel={trafficByLocation?.zoomLevel ?? 0}
              canZoomIn={trafficByLocation?.canZoomIn ?? false}
              focusTime={trafficByLocation?.focusTime}
              windowStart={trafficByLocation?.windowStart}
              windowEnd={trafficByLocation?.windowEnd}
              loading={dashboardLoading}
              onTimeSelect={handleAnalyticsZoom}
              onResetZoom={handleResetZoom}
              chartType="bar"
              legendPosition="top"
            />
            <OcclusionTrendsChart
              timeRange={timeRange}
              selectedDate={selectedDate}
              data={occlusionTrends?.series ?? []}
              bucketMinutes={occlusionTrends?.bucketMinutes ?? 60}
              zoomLevel={occlusionTrends?.zoomLevel ?? 0}
              canZoomIn={occlusionTrends?.canZoomIn ?? false}
              focusTime={occlusionTrends?.focusTime}
              windowStart={occlusionTrends?.windowStart}
              windowEnd={occlusionTrends?.windowEnd}
              loading={dashboardLoading}
              onTimeSelect={handleAnalyticsZoom}
              onResetZoom={handleResetZoom}
              chartType={inOutChartType}
              onChartTypeChange={setInOutChartType}
            />
          </div>
          <OcclusionMap
            hourFilter={hourFilter}
            onHourFilterChange={setHourFilter}
            data={occlusion}
            loading={dashboardLoading}
          />
        </div>
      </div>
    </div>
  )
}
