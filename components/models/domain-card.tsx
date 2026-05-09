"use client"

import { useMemo, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import {
  AlertCircle,
  CheckCircle2,
  Loader2,
  Trash2,
  Upload,
} from "lucide-react"
import {
  deleteDomainWeight,
  uploadDomainModel,
  updateModelSettings,
  type ModelDomain,
  type ModelDomainInfo,
  type ModelWeightInfo,
} from "@/lib/api"
import { formatBytes, formatDateTime } from "@/lib/format"

interface DomainCardProps {
  info: ModelDomainInfo
  onChanged: (next: ModelDomainInfo) => void
}

const FRAMEWORK_LABELS: Record<string, string> = {
  "ultralytics-yolov8": "Ultralytics YOLOv8",
  rtdetr: "RT-DETR (Occlusion-Robust)",
}

const DOMAIN_LABELS: Record<ModelDomain, string> = {
  pedestrian: "Pedestrian Detection",
  vehicle: "Vehicle Detection",
}

const DOMAIN_DESCRIPTIONS: Record<ModelDomain, string> = {
  pedestrian: "Active model used for pedestrian tracking and PTSI scoring",
  vehicle: "Active model used for vehicle gate counting and traffic LOS",
}

export function DomainCard({ info, onChanged }: DomainCardProps) {
  const [pendingActiveId, setPendingActiveId] = useState<string | null>(null)
  const [savingActive, setSavingActive] = useState(false)
  const [removingId, setRemovingId] = useState<string | null>(null)
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadLabel, setUploadLabel] = useState("")
  const [setActiveOnUpload, setSetActiveOnUpload] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const selectedActiveId = pendingActiveId ?? info.activeWeightId ?? ""
  const dirty = pendingActiveId !== null && pendingActiveId !== info.activeWeightId

  const orderedWeights = useMemo<ModelWeightInfo[]>(() => {
    return [...info.weights].sort((a, b) => {
      if (a.isSeed !== b.isSeed) return a.isSeed ? -1 : 1
      return a.filename.localeCompare(b.filename)
    })
  }, [info.weights])

  const activeMissing = info.active != null && info.active.exists === false

  function flashNotice(message: string) {
    setNotice(message)
    setError(null)
    window.setTimeout(() => setNotice((current) => (current === message ? null : current)), 4000)
  }

  async function handleSaveActive() {
    if (!pendingActiveId || pendingActiveId === info.activeWeightId) return
    setSavingActive(true)
    setError(null)
    try {
      const next = await updateModelSettings({ domain: info.domain, weightId: pendingActiveId })
      onChanged(next)
      setPendingActiveId(null)
      flashNotice("Active weight updated.")
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSavingActive(false)
    }
  }

  async function handleUpload() {
    if (!uploadFile) return
    setUploading(true)
    setError(null)
    try {
      const next = await uploadDomainModel(info.domain, uploadFile, {
        label: uploadLabel.trim() || undefined,
        setActive: setActiveOnUpload,
      })
      onChanged(next)
      setUploadFile(null)
      setUploadLabel("")
      flashNotice(`Uploaded ${uploadFile.name}.`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setUploading(false)
    }
  }

  async function handleRemove(weight: ModelWeightInfo) {
    if (weight.isSeed) return
    if (!window.confirm(`Remove ${weight.filename}? This deletes the file from disk.`)) return
    setRemovingId(weight.id)
    setError(null)
    try {
      const next = await deleteDomainWeight(info.domain, weight.id)
      onChanged(next)
      flashNotice(`Removed ${weight.filename}.`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRemovingId(null)
    }
  }

  return (
    <Card className="bg-card/60 border-border/60">
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle className="text-lg">{DOMAIN_LABELS[info.domain]}</CardTitle>
            <CardDescription>{DOMAIN_DESCRIPTIONS[info.domain]}</CardDescription>
          </div>
          <Badge variant="outline" className="capitalize">
            {info.domain}
          </Badge>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
          <Badge variant="secondary">{FRAMEWORK_LABELS[info.framework] ?? info.framework}</Badge>
          <Badge variant="secondary">tag: {info.ultralyticsTag}</Badge>
          {info.detectionClasses.length > 0 && (
            <Badge variant="secondary">classes: {info.detectionClasses.join(", ")}</Badge>
          )}
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {activeMissing && (
          <div className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-200">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
            <span>
              Active weight file is missing on disk: <code className="font-mono text-xs">{info.active?.absolutePath}</code>
            </span>
          </div>
        )}

        <section>
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">Available weights</Label>
          <RadioGroup
            value={selectedActiveId}
            onValueChange={(value) => setPendingActiveId(value)}
            className="mt-2 space-y-2"
          >
            {orderedWeights.map((weight) => {
              const isActive = info.activeWeightId === weight.id
              const isPending = pendingActiveId === weight.id && !isActive
              return (
                <div
                  key={weight.id}
                  className={`flex items-start gap-3 rounded-md border p-3 ${
                    isActive
                      ? "border-primary/60 bg-primary/5"
                      : isPending
                        ? "border-amber-500/40 bg-amber-500/5"
                        : "border-border/60 bg-secondary/20"
                  }`}
                >
                  <RadioGroupItem
                    id={`weight-${info.domain}-${weight.id}`}
                    value={weight.id}
                    className="mt-1"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <Label
                        htmlFor={`weight-${info.domain}-${weight.id}`}
                        className="font-medium text-sm cursor-pointer"
                      >
                        {weight.label ?? weight.filename}
                      </Label>
                      {isActive && (
                        <Badge variant="default" className="gap-1 text-[10px]">
                          <CheckCircle2 className="h-3 w-3" /> active
                        </Badge>
                      )}
                      {weight.isSeed && (
                        <Badge variant="outline" className="text-[10px]">seed</Badge>
                      )}
                      {!weight.exists && (
                        <Badge variant="destructive" className="text-[10px]">missing</Badge>
                      )}
                    </div>
                    <div className="mt-1 grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs text-muted-foreground">
                      <span className="font-mono truncate">{weight.relativePath}</span>
                      <span>{formatBytes(weight.sizeBytes)}</span>
                      {weight.uploadedAt && (
                        <span className="col-span-2">uploaded {formatDateTime(weight.uploadedAt)}</span>
                      )}
                    </div>
                  </div>
                  {!weight.isSeed && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-muted-foreground hover:text-destructive"
                      disabled={removingId === weight.id}
                      onClick={() => handleRemove(weight)}
                      title="Remove weight"
                    >
                      {removingId === weight.id ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                    </Button>
                  )}
                </div>
              )
            })}
          </RadioGroup>

          <div className="mt-3 flex items-center justify-end gap-2">
            {dirty && (
              <span className="text-xs text-amber-300">
                Unsaved selection — click Apply to switch the active weight.
              </span>
            )}
            <Button
              size="sm"
              disabled={!dirty || savingActive}
              onClick={handleSaveActive}
            >
              {savingActive && <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />}
              Apply
            </Button>
          </div>
        </section>

        <section className="rounded-md border border-border/60 bg-secondary/10 p-4">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground">Upload new weights</Label>
          <div className="mt-3 grid gap-3">
            <Input
              type="file"
              accept={info.domain === "pedestrian" ? ".pt" : ".pt,.pth"}
              onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)}
            />
            <Input
              type="text"
              placeholder="Optional label (e.g. v2 fine-tune)"
              value={uploadLabel}
              onChange={(event) => setUploadLabel(event.target.value)}
            />
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Switch
                  id={`set-active-${info.domain}`}
                  checked={setActiveOnUpload}
                  onCheckedChange={setSetActiveOnUpload}
                />
                <Label htmlFor={`set-active-${info.domain}`} className="text-sm">
                  Set as active after upload
                </Label>
              </div>
              <Button
                size="sm"
                disabled={!uploadFile || uploading}
                onClick={handleUpload}
              >
                {uploading ? (
                  <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Upload className="mr-2 h-3.5 w-3.5" />
                )}
                Upload
              </Button>
            </div>
          </div>
        </section>

        {error && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
        {notice && (
          <div className="flex items-start gap-2 rounded-md border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-200">
            <CheckCircle2 className="h-4 w-4 mt-0.5 shrink-0" />
            <span>{notice}</span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
