"use client"

import Link from "next/link"
import { Video, Wifi } from "lucide-react"

interface VideoThumbnailProps {
  id: string
  label?: string
  location: string
  timestamp: string
  date: string
  startTime?: string
  pedestrianCount: number
  mediaUrl?: string | null
  rawPath?: string | null
  processedPath?: string | null
}

export function VideoThumbnail({ id, label, location, timestamp, date, startTime, pedestrianCount, mediaUrl, rawPath, processedPath }: VideoThumbnailProps) {
  const statusLabel = rawPath ? "Uploaded" : "Sample"
  const previewLabel = processedPath ? "Processed view" : rawPath ? "Raw view" : "Preview unavailable"

  return (
    <Link
      href={`/video/${id}`}
      className="group relative rounded-2xl overflow-hidden bg-secondary border border-border hover:border-primary/50 transition-all shadow-elevated-sm"
      style={{ aspectRatio: "16/10" }}
    >
      {mediaUrl ? (
        <video key={mediaUrl} src={mediaUrl} muted playsInline preload="metadata" className="absolute inset-0 h-full w-full object-cover bg-black" />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-slate-700 via-slate-800 to-slate-900 text-white/80">
          <div className="text-center">
            <Video className="mx-auto mb-2 h-7 w-7" />
            <p className="text-xs font-medium">No media preview</p>
          </div>
        </div>
      )}

      <div className="absolute inset-0 bg-gradient-to-t from-black/85 via-black/20 to-black/10" />

      {/* Status Indicators */}
      {!processedPath && (
        <div className="absolute top-3 left-3 flex items-center gap-2">
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-black/50 backdrop-blur-sm">
            <Wifi className="w-3 h-3 text-accent" />
            <span className="text-[10px] text-white font-medium">{statusLabel}</span>
          </div>
        </div>
      )}

      {/* Vehicle Count */}
      <div className={`absolute top-3 flex items-center gap-2 ${processedPath ? "left-3" : "right-3"}`}>
        <span className="text-xs font-bold text-white bg-primary/80 backdrop-blur-sm px-2.5 py-1 rounded-full">
          {pedestrianCount} vehicles
        </span>
      </div>

      {/* Bottom Info */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent p-4">
        <p className="text-sm font-semibold text-white truncate">{label ?? location}</p>
        <p className="text-xs text-white/70">{date} • {timestamp}</p>
        <p className="mt-1 text-[11px] text-white/60">{previewLabel}</p>
      </div>

      {/* Hover Overlay */}
      <div className="absolute inset-0 bg-black/10 opacity-0 group-hover:opacity-100 transition-opacity" />
    </Link>
  )
}
