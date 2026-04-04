"use client"

import { Suspense, useEffect, useMemo, useState } from "react"
import { useSearchParams } from "next/navigation"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { AlertCircle, ArrowLeft, Calendar, Clock, Loader2, MapPin, Play, ScanSearch, Sparkles, UserRound, Video } from "lucide-react"
import { useLoading } from "@/components/ui/walking-loader"
import { getMediaUrl, searchVideos, type SearchResult } from "@/lib/api"

function normalizeResultText(value?: string | null) {
  return (value ?? "").replace(/\s+/g, " ").trim().toLowerCase()
}

function semanticPercent(value?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null
  }
  return `${Math.round(value * 100)}%`
}

function resultBadgeLabel(result: SearchResult) {
  if (result.matchStrategy === "semantic") {
    return result.possibleMatch ? "Possible semantic match" : "Semantic match"
  }
  if (result.matchStrategy === "event") {
    return "Event match"
  }
  return "AI match"
}

function SearchContent() {
  const searchParams = useSearchParams()
  const query = searchParams.get("q")?.trim() ?? ""
  const [results, setResults] = useState<SearchResult[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const { showLoader, updateLoader, hideLoader } = useLoading()

  useEffect(() => {
    let cancelled = false
    let stageTimer: ReturnType<typeof setTimeout> | null = null

    const runSearch = async () => {
      setLoading(true)
      setError(null)
      showLoader("ALIVE is searching for matching pedestrians...")

      stageTimer = setTimeout(() => {
        if (!cancelled) {
          updateLoader({ label: "Comparing representative pedestrian crops and ranking likely matches..." })
        }
      }, 900)

      try {
        const response = await searchVideos(query)
        if (!cancelled) {
          setResults(response)
        }
      } catch (error) {
        if (!cancelled) {
          setError(error instanceof Error ? error.message : "Failed to run search.")
        }
      } finally {
        if (stageTimer) {
          clearTimeout(stageTimer)
        }
        if (!cancelled) {
          setLoading(false)
          hideLoader()
        }
      }
    }

    void runSearch()

    return () => {
      cancelled = true
      if (stageTimer) {
        clearTimeout(stageTimer)
      }
      hideLoader()
    }
  }, [hideLoader, query, showLoader, updateLoader])

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-border bg-card">
        <div className="flex items-center gap-4">
          <Link href="/">
            <Button variant="ghost" size="icon" className="text-muted-foreground hover:text-foreground">
              <ArrowLeft className="w-5 h-5" />
            </Button>
          </Link>
          <div>
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-accent" />
              <h1 className="text-xl font-semibold text-foreground">AI Search Results</h1>
            </div>
            <p className="text-sm text-muted-foreground mt-0.5">
              Query: {'"'}{query || "recent activity"}{'"'}
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">
            {loading ? "Searching..." : `${results.length} matches found`}
          </span>
        </div>
      </header>

      {/* Search Results */}
      <div className="flex-1 overflow-auto p-6">
        <div className="max-w-4xl mx-auto space-y-4">
          {/* Results Header */}
          <div className="flex items-center justify-between mb-6">
            <p className="text-sm text-muted-foreground">
              {query
                ? `Showing top ${results.length} snippets that match your search`
                : `Showing ${results.length} recent snippets from the local event feed`}
            </p>
          </div>

          {loading ? (
            <div className="flex items-center justify-center rounded-2xl border border-border bg-card p-12 text-muted-foreground">
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Loading search results...
            </div>
          ) : error ? (
            <div className="flex items-start gap-3 rounded-2xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          ) : results.length > 0 ? (
            results.map((result, index) => (
              <SearchResultCard 
                key={result.id} 
                result={result} 
                index={index + 1}
              />
            ))
          ) : (
            <div className="rounded-2xl border border-border bg-card p-12 text-center">
              <p className="text-lg font-medium text-foreground">No results found</p>
              <p className="mt-2 text-sm text-muted-foreground">
                Try a different AI search query or upload more local footage first.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function SearchResultCard({ 
  result, 
  index 
}: { 
  result: SearchResult
  index: number 
}) {
  const [thumbnailFailed, setThumbnailFailed] = useState(false)
  const [previewFailed, setPreviewFailed] = useState(false)
  const thumbnailUrl = useMemo(() => getMediaUrl(result.thumbnailPath), [result.thumbnailPath])
  const previewUrl = useMemo(() => getMediaUrl(result.previewPath), [result.previewPath])
  const hasVisionDetails = Boolean(
    result.visualSummary ||
      result.visualLabels?.length ||
      result.visualObjects?.length ||
      result.visualLogos?.length ||
      result.visualText?.length,
  )
  const normalizedMatchReason = normalizeResultText(result.matchReason)
  const normalizedAppearanceSummary = normalizeResultText(result.appearanceSummary)
  const showAppearanceSummary = Boolean(
    result.appearanceSummary && normalizedAppearanceSummary && normalizedAppearanceSummary !== normalizedMatchReason,
  )
  const showVisualSummary = Boolean(result.visualSummary && !result.appearanceSummary?.includes(result.visualSummary))
  const footageHref = useMemo(() => {
    if (typeof result.offsetSeconds === "number") {
      return `/video/${result.videoId}?seek=${encodeURIComponent(String(result.offsetSeconds))}`
    }
    return `/video/${result.videoId}`
  }, [result.offsetSeconds, result.videoId])
  const showImage = Boolean(thumbnailUrl && !thumbnailFailed)
  const showPreviewVideo = Boolean(!showImage && previewUrl && !previewFailed)
  const semanticScoreText = semanticPercent(result.semanticScore)
  const badgeLabel = resultBadgeLabel(result)
  const badgeClassName = result.matchStrategy === "semantic"
    ? "bg-cyan-500/90 text-white"
    : result.matchStrategy === "event"
      ? "bg-amber-500/90 text-black"
      : "bg-primary/90 text-primary-foreground"

  return (
    <div className="flex gap-4 p-4 rounded-xl bg-card border border-border hover:border-primary/30 transition-all group">
      {/* Thumbnail */}
      <Link href={footageHref} className="relative block w-64 aspect-video rounded-lg overflow-hidden bg-secondary shrink-0 focus:outline-none focus:ring-2 focus:ring-primary/60">
        {showImage ? (
          <img
            src={thumbnailUrl ?? undefined}
            alt={`Pedestrian match ${result.pedestrianId ?? index}`}
            className="absolute inset-0 h-full w-full object-cover"
            onError={() => setThumbnailFailed(true)}
          />
        ) : showPreviewVideo ? (
          <video
            key={previewUrl}
            src={previewUrl ?? undefined}
            muted
            playsInline
            preload="metadata"
            className="absolute inset-0 h-full w-full object-cover bg-black"
            onError={() => setPreviewFailed(true)}
          />
        ) : (
          <div className="absolute inset-0 bg-gradient-to-br from-slate-800 via-slate-900 to-slate-950">
            <div 
              className="absolute inset-0 opacity-20"
              style={{
                backgroundImage: `linear-gradient(rgba(255,255,255,.03) 1px, transparent 1px),
                                  linear-gradient(90deg, rgba(255,255,255,.03) 1px, transparent 1px)`,
                backgroundSize: '15px 15px'
              }}
            />
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-300">
              <Video className="h-6 w-6" />
              <span className="text-xs">Preview unavailable</span>
            </div>
          </div>
        )}

        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-3">
          <div className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-[10px] font-medium uppercase tracking-wide ${badgeClassName}`}>
            <ScanSearch className="h-3 w-3" />
            {badgeLabel}
          </div>
        </div>

        {/* Play overlay */}
        <div className="absolute inset-0 flex items-center justify-center bg-black/30 opacity-0 transition-opacity group-hover:opacity-100">
          <div className="w-12 h-12 rounded-full bg-primary/90 flex items-center justify-center shadow-lg">
            <Play className="w-5 h-5 text-primary-foreground ml-0.5" />
          </div>
        </div>

        {/* Result number */}
        <div className="absolute top-2 left-2">
          <span className="text-xs font-medium bg-accent text-accent-foreground px-2 py-1 rounded">
            #{index}
          </span>
        </div>
      </Link>

      {/* Content */}
      <div className="flex-1 flex flex-col">
        <div className="flex items-start justify-between mb-2">
          <div>
            <h3 className="text-lg font-medium text-foreground">{result.location}</h3>
            <p className="text-sm text-primary">{result.matchReason}</p>
            {result.matchStrategy === "semantic" ? (
              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                {semanticScoreText ? (
                  <span className="rounded-full bg-cyan-500/10 px-2.5 py-1 font-medium text-cyan-700 dark:text-cyan-300">
                    Semantic score {semanticScoreText}
                  </span>
                ) : null}
                {result.possibleMatch ? (
                  <span className="rounded-full bg-amber-500/10 px-2.5 py-1 font-medium text-amber-700 dark:text-amber-300">
                    Lower-confidence appearance match
                  </span>
                ) : null}
              </div>
            ) : null}
            {showAppearanceSummary ? (
              <p className="mt-2 text-sm text-muted-foreground">{result.appearanceSummary}</p>
            ) : null}
            {hasVisionDetails ? (
              <div className="mt-3 rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-emerald-700 dark:text-emerald-400">
                  Cloud Vision noticed additional details
                </p>
                {showVisualSummary ? (
                  <p className="mt-1 text-sm text-foreground">{result.visualSummary}</p>
                ) : null}
                <div className="mt-2 flex flex-wrap gap-2 text-xs">
                  {(result.visualObjects ?? []).map((item) => (
                    <span key={`object-${item}`} className="rounded-full bg-emerald-500/10 px-2.5 py-1 text-emerald-700 dark:text-emerald-300">
                      Object: {item}
                    </span>
                  ))}
                  {(result.visualLabels ?? []).map((item) => (
                    <span key={`label-${item}`} className="rounded-full bg-primary/10 px-2.5 py-1 text-primary">
                      Label: {item}
                    </span>
                  ))}
                  {(result.visualLogos ?? []).map((item) => (
                    <span key={`logo-${item}`} className="rounded-full bg-amber-500/10 px-2.5 py-1 text-amber-700 dark:text-amber-300">
                      Logo: {item}
                    </span>
                  ))}
                  {(result.visualText ?? []).map((item) => (
                    <span key={`text-${item}`} className="rounded-full bg-secondary px-2.5 py-1 text-foreground">
                      Text: {item}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
          <div className="flex shrink-0 flex-col items-end gap-2">
            <div className="flex items-center gap-1 rounded-full bg-primary/10 px-2 py-1 whitespace-nowrap">
              <span className="text-xs font-medium text-primary whitespace-nowrap">{badgeLabel}</span>
            </div>
            <div className="rounded-full bg-secondary px-2.5 py-1 text-xs text-foreground">
              Confidence {Math.max(0, Math.min(100, Math.round(result.confidence)))}%
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4 text-sm text-muted-foreground mb-4">
          <div className="flex items-center gap-1.5">
            <Clock className="w-4 h-4" />
            {result.timestamp}
          </div>
          <div className="flex items-center gap-1.5">
            <Calendar className="w-4 h-4" />
            {result.date}
          </div>
          <div className="flex items-center gap-1.5">
            <MapPin className="w-4 h-4" />
            {result.location}
          </div>
          {typeof result.pedestrianId === "number" ? (
            <div className="flex items-center gap-1.5">
              <UserRound className="w-4 h-4" />
              Track #{result.pedestrianId}
            </div>
          ) : null}
          {typeof result.frame === "number" ? (
            <div className="rounded-full bg-secondary px-2 py-1 text-xs text-foreground">
              Frame {result.frame}
            </div>
          ) : null}
        </div>

        <div className="mt-auto">
          <Link href={footageHref}>
            <Button className="bg-primary text-primary-foreground hover:bg-primary/90">
              <Play className="w-4 h-4 mr-2" />
              View Footage
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}

export default function SearchPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
      </div>
    }>
      <SearchContent />
    </Suspense>
  )
}
