"use client"

import { useState, useEffect } from "react"
import { uploadModel, getCurrentModel, getInferenceStatus, type ModelInfo, type InferenceStatus } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Upload, Settings, ShieldAlert, CheckCircle2, Box } from "lucide-react"

function ModelConfigCard({
  title,
  description,
  pipeline,
}: {
  title: string
  description: string
  pipeline: string
}) {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [inferenceStatus, setInferenceStatus] = useState<InferenceStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const fetchData = async () => {
    try {
      const [info, status] = await Promise.all([
        getCurrentModel(pipeline),
        getInferenceStatus(pipeline),
      ])
      setModelInfo(info)
      setInferenceStatus(status)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void fetchData()
  }, [pipeline])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    setError(null)
    setSuccess(null)

    try {
      await uploadModel(file, pipeline)
      setSuccess("Model uploaded successfully!")
      await fetchData()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload model.")
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="bg-card/50 backdrop-blur-sm border border-border rounded-2xl p-6 shadow-sm">
      <div className="flex items-start justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-foreground flex items-center gap-2">
            <Box className="w-5 h-5 text-primary" />
            {title}
          </h2>
          <p className="text-sm text-muted-foreground mt-1">{description}</p>
        </div>
        <div className="bg-secondary/50 rounded-full px-3 py-1 text-xs font-medium text-muted-foreground border border-border flex items-center gap-2">
          {inferenceStatus?.ready ? (
            <><CheckCircle2 className="w-3 h-3 text-green-500" /> Ready</>
          ) : (
            <><ShieldAlert className="w-3 h-3 text-yellow-500" /> Needs Configuration</>
          )}
        </div>
      </div>

      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-secondary/30 rounded-xl p-4 border border-border/50">
            <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Current Model</div>
            <div className="font-medium text-foreground truncate" title={modelInfo?.currentModel ?? "None"}>
              {modelInfo?.currentModel ?? "None"}
            </div>
          </div>
          <div className="bg-secondary/30 rounded-xl p-4 border border-border/50">
            <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Last Updated</div>
            <div className="font-medium text-foreground">
              {modelInfo?.uploadedAt ? new Date(modelInfo.uploadedAt).toLocaleDateString() : "Never"}
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-destructive/10 border border-destructive/30 text-destructive text-sm p-3 rounded-xl flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 shrink-0" />
            {error}
          </div>
        )}

        {success && (
          <div className="bg-green-500/10 border border-green-500/30 text-green-500 text-sm p-3 rounded-xl flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            {success}
          </div>
        )}

        <div className="pt-2">
          <div className="relative">
            <input
              type="file"
              accept=".pt,.pth"
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
              onChange={handleUpload}
              disabled={uploading || loading}
            />
            <Button 
              className="w-full bg-primary text-primary-foreground hover:bg-primary/90 rounded-xl shadow-elevated-sm pointer-events-none"
              disabled={uploading || loading}
            >
              {uploading ? (
                "Uploading..."
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Upload New Model Weights (.pt / .pth)
                </>
              )}
            </Button>
          </div>
          <p className="text-xs text-center text-muted-foreground mt-3">
            Supported formats: PyTorch weights (.pt, .pth)
          </p>
        </div>
      </div>
    </div>
  )
}

export default function SettingsPage() {
  return (
    <div className="flex h-full">
      <div className="flex-1 flex flex-col overflow-auto">
        {/* Header */}
        <header className="flex items-center px-6 py-4 border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
          <div className="w-10 h-10 rounded-xl bg-secondary flex items-center justify-center mr-4 shadow-sm border border-border">
            <Settings className="w-5 h-5 text-foreground" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-foreground">Model Settings</h1>
            <p className="text-sm text-muted-foreground">Configure AI models for detection pipelines</p>
          </div>
        </header>

        {/* Content */}
        <div className="p-6 max-w-5xl mx-auto w-full">
          <div className="grid md:grid-cols-2 gap-6">
            <ModelConfigCard 
              title="Pedestrian Pipeline"
              description="Manages YOLOv8 models for tracking people, detecting loitering, and analyzing crowd density."
              pipeline="pedestrian"
            />
            <ModelConfigCard 
              title="Vehicle Pipeline"
              description="Manages Occlusion-Robust RTDETR models for traffic counting, vehicle detection, and congestion analysis."
              pipeline="vehicle"
            />
          </div>
        </div>
      </div>
    </div>
  )
}
