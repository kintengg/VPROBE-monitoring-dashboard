"use client"

import { useCallback, useEffect, useState } from "react"
import { Loader2, RefreshCw, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { DomainCard } from "@/components/models/domain-card"
import {
  getModelRegistry,
  type ModelDomainInfo,
  type ModelRegistryResponse,
} from "@/lib/api"

export default function ModelSetupPage() {
  const [registry, setRegistry] = useState<ModelRegistryResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getModelRegistry()
      setRegistry(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  function handleDomainChanged(updated: ModelDomainInfo) {
    setRegistry((current) => {
      if (!current) return current
      return {
        ...current,
        domains: current.domains.map((d) => (d.domain === updated.domain ? updated : d)),
      }
    })
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-8">
      <header className="mb-8 flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs uppercase tracking-[0.28em] text-muted-foreground">
            <Sparkles className="h-3.5 w-3.5" />
            Model Setup
          </div>
          <h1 className="mt-2 text-3xl font-semibold">Detection Model Configuration</h1>
          <p className="mt-2 text-sm text-muted-foreground max-w-2xl">
            Manage the active weights for pedestrian and vehicle inference. Uploaded weights
            land under <code className="font-mono text-xs">backend/storage/models/&lt;domain&gt;/</code> and become available
            for hot-swap on the next upload run.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => void refresh()} disabled={loading}>
          {loading ? (
            <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
          ) : (
            <RefreshCw className="mr-2 h-3.5 w-3.5" />
          )}
          Refresh
        </Button>
      </header>

      {error && (
        <div className="mb-6 rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {!registry && loading && (
        <div className="rounded-md border border-border/60 bg-secondary/20 p-12 text-center text-muted-foreground">
          <Loader2 className="mx-auto mb-3 h-6 w-6 animate-spin" />
          Loading registry…
        </div>
      )}

      {registry && (
        <div className="grid gap-6 md:grid-cols-2">
          {registry.domains.map((domainInfo) => (
            <DomainCard
              key={domainInfo.domain}
              info={domainInfo}
              onChanged={handleDomainChanged}
            />
          ))}
        </div>
      )}
    </div>
  )
}
