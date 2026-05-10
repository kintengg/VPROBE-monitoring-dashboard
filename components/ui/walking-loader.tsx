"use client"

import { X } from "lucide-react"
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react"

interface WalkingLoaderProps {
  isVisible: boolean
  label: string
  progress?: number | null
  onClose?: () => void
  variant?: "pedestrian" | "vehicle"
}

export function WalkingLoader({ isVisible, label, progress = null, onClose, variant = "pedestrian" }: WalkingLoaderProps) {
  const [frame, setFrame] = useState(0)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  
  useEffect(() => {
    if (!isVisible) return
    
    const interval = setInterval(() => {
      setFrame((prev) => (prev + 1) % 4)
    }, 200)
    
    return () => clearInterval(interval)
  }, [isVisible])

  useEffect(() => {
    if (!isVisible) {
      setElapsedSeconds(0)
      return
    }

    const startedAt = Date.now()
    setElapsedSeconds(0)

    const interval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000))
    }, 1000)

    return () => clearInterval(interval)
  }, [isVisible])

  if (!isVisible) return null

  const showElapsedTime = elapsedSeconds > 5

  // Walking stick figure frames
  const getStickFigure = (frameNum: number) => {
    const poses = [
      // Frame 0 - Right leg forward, left arm forward
      {
        leftArm: "M24 28 L18 36",
        rightArm: "M36 28 L42 34",
        leftLeg: "M26 48 L22 62 L18 68",
        rightLeg: "M34 48 L40 60 L44 68",
        body: "M30 28 L30 48",
        head: { cx: 30, cy: 20 }
      },
      // Frame 1 - Legs passing, arms neutral
      {
        leftArm: "M24 28 L20 38",
        rightArm: "M36 28 L40 38",
        leftLeg: "M26 48 L28 62 L28 68",
        rightLeg: "M34 48 L32 62 L32 68",
        body: "M30 28 L30 48",
        head: { cx: 30, cy: 20 }
      },
      // Frame 2 - Left leg forward, right arm forward
      {
        leftArm: "M24 28 L18 34",
        rightArm: "M36 28 L42 36",
        leftLeg: "M26 48 L20 60 L16 68",
        rightLeg: "M34 48 L38 62 L42 68",
        body: "M30 28 L30 48",
        head: { cx: 30, cy: 20 }
      },
      // Frame 3 - Legs passing other way
      {
        leftArm: "M24 28 L20 38",
        rightArm: "M36 28 L40 38",
        leftLeg: "M26 48 L32 62 L32 68",
        rightLeg: "M34 48 L28 62 L28 68",
        body: "M30 28 L30 48",
        head: { cx: 30, cy: 20 }
      },
    ]
    return poses[frameNum]
  }

  const pose = getStickFigure(frame)

  return (
    <div className="fixed inset-0 z-[80] flex items-start justify-center pt-32 pointer-events-none">
      <div className="pointer-events-auto relative flex w-[min(92vw,32rem)] flex-col items-center gap-4 rounded-3xl border border-border bg-card p-6 shadow-elevated">
        {onClose ? (
          <button
            type="button"
            onClick={onClose}
            className="absolute right-4 top-4 inline-flex h-8 w-8 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
            aria-label="Hide loader"
          >
            <X className="h-4 w-4" />
          </button>
        ) : null}
        {/* Walking Animation Container */}
        <div className="relative w-20 h-24 flex items-center justify-center">
          {/* Ground line with moving dots */}
          <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-border overflow-hidden">
            <div 
              className="absolute top-0 left-0 w-full h-full"
              style={{
                background: 'repeating-linear-gradient(90deg, transparent, transparent 8px, var(--primary) 8px, var(--primary) 12px)',
                animation: 'slideLeft 0.4s linear infinite',
              }}
            />
          </div>
          
          {variant === "vehicle" ? (
            <svg
              viewBox="0 0 80 56"
              className="w-20 h-16"
              style={{
                filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.3))',
                transform: `translateY(${frame % 2 === 0 ? '-1px' : '1px'})`,
                transition: 'transform 0.1s linear',
              }}
            >
              {/* Car body — silhouette of a sedan */}
              <path
                d="M8 38 Q8 30 16 28 L24 18 Q26 14 32 14 L52 14 Q58 14 60 18 L68 28 Q72 30 72 38 L72 42 Q72 44 70 44 L62 44 L58 44 Q58 40 54 40 Q50 40 50 44 L30 44 Q30 40 26 40 Q22 40 22 44 L18 44 L10 44 Q8 44 8 42 Z"
                stroke="url(#gradient)"
                strokeWidth="2.2"
                strokeLinejoin="round"
                fill="none"
              />
              {/* Windows */}
              <path
                d="M26 28 L30 18 L50 18 L54 28 Z"
                stroke="url(#gradient)"
                strokeWidth="1.6"
                strokeLinejoin="round"
                fill="none"
              />
              {/* Front + rear wheels */}
              <circle cx="26" cy="44" r="5" stroke="url(#gradient)" strokeWidth="2" fill="none" />
              <circle cx="54" cy="44" r="5" stroke="url(#gradient)" strokeWidth="2" fill="none" />
              {/* Headlight (subtle pulse via frame) */}
              <circle
                cx="68"
                cy="34"
                r={frame % 2 === 0 ? 1.6 : 2.2}
                fill="#22D3EE"
                opacity={frame % 2 === 0 ? 0.6 : 1}
              />

              <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#A855F7" />
                  <stop offset="100%" stopColor="#22D3EE" />
                </linearGradient>
              </defs>
            </svg>
          ) : (
            <svg
              viewBox="0 0 60 72"
              className="w-16 h-20"
              style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.3))' }}
            >
              {/* Head */}
              <circle
                cx={pose.head.cx}
                cy={pose.head.cy}
                r="8"
                fill="none"
                stroke="url(#gradient)"
                strokeWidth="3"
                strokeLinecap="round"
              />

              {/* Body */}
              <path
                d={pose.body}
                stroke="url(#gradient)"
                strokeWidth="3"
                strokeLinecap="round"
                fill="none"
              />

              {/* Arms */}
              <path
                d={pose.leftArm}
                stroke="url(#gradient)"
                strokeWidth="2.5"
                strokeLinecap="round"
                fill="none"
              />
              <path
                d={pose.rightArm}
                stroke="url(#gradient)"
                strokeWidth="2.5"
                strokeLinecap="round"
                fill="none"
              />

              {/* Legs */}
              <path
                d={pose.leftLeg}
                stroke="url(#gradient)"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />
              <path
                d={pose.rightLeg}
                stroke="url(#gradient)"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />

              {/* Gradient Definition */}
              <defs>
                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#A855F7" />
                  <stop offset="100%" stopColor="#22D3EE" />
                </linearGradient>
              </defs>
            </svg>
          )}
        </div>
        
        {/* Label */}
        <div className="w-full max-w-[26rem] text-center">
          <p className="break-words text-sm font-medium leading-snug text-foreground">{label}</p>
          {typeof progress === "number" ? (
            <p className="mt-2 text-xs font-medium uppercase tracking-[0.18em] text-primary">{progress}% complete</p>
          ) : null}
          <div className="mt-1 flex items-center justify-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="h-1.5 w-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
          {showElapsedTime ? (
            <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
              Elapsed time: <span className="font-medium text-foreground">{elapsedSeconds} {elapsedSeconds === 1 ? "second" : "seconds"}</span>
            </p>
          ) : null}
        </div>
      </div>
      
      {/* CSS Animation */}
      <style jsx>{`
        @keyframes slideLeft {
          from {
            transform: translateX(0);
          }
          to {
            transform: translateX(-20px);
          }
        }
      `}</style>
    </div>
  )
}

interface LoadingContextType {
  showLoader: (label: string, progress?: number | null) => void
  updateLoader: (payload: { label?: string; progress?: number | null }) => void
  hideLoader: () => void
  isLoading: boolean
  loadingLabel: string
  loadingProgress: number | null
}

const LoadingContext = createContext<LoadingContextType | null>(null)

export function LoadingProvider({ children }: { children: ReactNode }) {
  const [isLoading, setIsLoading] = useState(false)
  const [loadingLabel, setLoadingLabel] = useState("")
  const [loadingProgress, setLoadingProgress] = useState<number | null>(null)

  const showLoader = useCallback((label: string, progress: number | null = null) => {
    setLoadingLabel(label)
    setLoadingProgress(progress)
    setIsLoading(true)
  }, [])

  const updateLoader = useCallback(({ label, progress }: { label?: string; progress?: number | null }) => {
    if (typeof label === "string") {
      setLoadingLabel(label)
    }
    if (progress !== undefined) {
      setLoadingProgress(progress)
    }
  }, [])

  const hideLoader = useCallback(() => {
    setIsLoading(false)
    setLoadingLabel("")
    setLoadingProgress(null)
  }, [])

  const value = useMemo(
    () => ({ showLoader, updateLoader, hideLoader, isLoading, loadingLabel, loadingProgress }),
    [hideLoader, isLoading, loadingLabel, loadingProgress, showLoader, updateLoader],
  )

  return (
    <LoadingContext.Provider value={value}>
      {children}
      <WalkingLoader isVisible={isLoading} label={loadingLabel} progress={loadingProgress} />
    </LoadingContext.Provider>
  )
}

export function useLoading() {
  const context = useContext(LoadingContext)
  if (!context) {
    throw new Error("useLoading must be used within a LoadingProvider")
  }
  return context
}
