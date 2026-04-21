"use client"

import { useEffect, useId, useRef, useState } from "react"

import { cn } from "@/lib/utils"

const MAX_PUPIL_OFFSET = { x: 2.8, y: 2.2 }
const TRACKING_DISTANCE = 240
const EYE_CENTER_Y = 72

function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(false)

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") {
      return
    }

    const mediaQuery = window.matchMedia(query)
    const handleChange = () => setMatches(mediaQuery.matches)

    handleChange()
    mediaQuery.addEventListener("change", handleChange)
    return () => mediaQuery.removeEventListener("change", handleChange)
  }, [query])

  return matches
}

export function BantaySidebarLogo({ className }: { className?: string }) {
  const logoId = useId().replace(/:/g, "")
  const rootRef = useRef<HTMLDivElement | null>(null)
  const currentOffsetRef = useRef({ x: 0, y: 0 })
  const targetOffsetRef = useRef({ x: 0, y: 0 })
  const [pupilOffset, setPupilOffset] = useState({ x: 0, y: 0 })
  const [isBlinking, setIsBlinking] = useState(false)
  const [isHovered, setIsHovered] = useState(false)

  const prefersReducedMotion = useMediaQuery("(prefers-reduced-motion: reduce)")
  const canTrackCursor = useMediaQuery("(hover: hover) and (pointer: fine)")
  const shouldTrackCursor = canTrackCursor && !prefersReducedMotion

  useEffect(() => {
    if (prefersReducedMotion) {
      setIsBlinking(false)
      return
    }

    let cancelled = false
    let blinkTimer = 0
    let reopenTimer = 0

    const queueBlink = () => {
      blinkTimer = window.setTimeout(() => {
        setIsBlinking(true)
        reopenTimer = window.setTimeout(() => {
          setIsBlinking(false)
          if (!cancelled) {
            queueBlink()
          }
        }, 170)
      }, 2600 + Math.random() * 2600)
    }

    queueBlink()
    return () => {
      cancelled = true
      window.clearTimeout(blinkTimer)
      window.clearTimeout(reopenTimer)
    }
  }, [prefersReducedMotion])

  useEffect(() => {
    if (!shouldTrackCursor || typeof window === "undefined") {
      targetOffsetRef.current = { x: 0, y: 0 }
      currentOffsetRef.current = { x: 0, y: 0 }
      setPupilOffset({ x: 0, y: 0 })
      return
    }

    let frameId = 0

    const updateTargetOffset = (event: PointerEvent) => {
      const bounds = rootRef.current?.getBoundingClientRect()
      if (!bounds) {
        return
      }

      const centerX = bounds.left + (bounds.width / 2)
      const centerY = bounds.top + (bounds.height / 2)
      const deltaX = event.clientX - centerX
      const deltaY = event.clientY - centerY
      const distance = Math.hypot(deltaX, deltaY)

      if (distance <= 0.001) {
        targetOffsetRef.current = { x: 0, y: 0 }
        return
      }

      const strength = Math.min(distance / TRACKING_DISTANCE, 1)
      targetOffsetRef.current = {
        x: (deltaX / distance) * MAX_PUPIL_OFFSET.x * strength,
        y: (deltaY / distance) * MAX_PUPIL_OFFSET.y * strength,
      }
    }

    const resetTargetOffset = () => {
      targetOffsetRef.current = { x: 0, y: 0 }
    }

    const animate = () => {
      const current = currentOffsetRef.current
      const target = targetOffsetRef.current
      const next = {
        x: current.x + ((target.x - current.x) * 0.18),
        y: current.y + ((target.y - current.y) * 0.18),
      }

      currentOffsetRef.current = next
      setPupilOffset((previous) => {
        if (Math.abs(previous.x - next.x) < 0.01 && Math.abs(previous.y - next.y) < 0.01) {
          return previous
        }
        return next
      })

      frameId = window.requestAnimationFrame(animate)
    }

    window.addEventListener("pointermove", updateTargetOffset, { passive: true })
    window.addEventListener("blur", resetTargetOffset)
    document.addEventListener("visibilitychange", resetTargetOffset)
    frameId = window.requestAnimationFrame(animate)

    return () => {
      window.removeEventListener("pointermove", updateTargetOffset)
      window.removeEventListener("blur", resetTargetOffset)
      document.removeEventListener("visibilitychange", resetTargetOffset)
      window.cancelAnimationFrame(frameId)
    }
  }, [shouldTrackCursor])

  const eyeScaleY = isBlinking ? 0.14 : 1
  const collarTextOpacity = isHovered ? 0.96 : 0.82

  return (
    <div
      ref={rootRef}
      className={cn(
        "relative flex h-full w-full items-center justify-center rounded-[1.75rem] transition-transform duration-300 ease-out",
        !prefersReducedMotion && "motion-safe:duration-500",
        !prefersReducedMotion && isHovered && "-translate-y-0.5",
        !prefersReducedMotion && !shouldTrackCursor && "bantay-logo-idle",
        className,
      )}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      role="img"
      aria-label="Bantay mascot logo"
    >
      <svg viewBox="0 0 160 160" className="h-full w-full overflow-visible">
        <defs>
          <linearGradient id={`${logoId}-ear`} x1="25" x2="50" y1="28" y2="96" gradientUnits="userSpaceOnUse">
            <stop offset="0" stopColor="#5d5a67" />
            <stop offset="0.52" stopColor="#403d49" />
            <stop offset="1" stopColor="#252938" />
          </linearGradient>
          <linearGradient id={`${logoId}-shell`} x1="24" x2="136" y1="24" y2="132" gradientUnits="userSpaceOnUse">
            <stop offset="0" stopColor="#ffffff" />
            <stop offset="0.36" stopColor="#e4ebf4" />
            <stop offset="1" stopColor="#98a4b4" />
          </linearGradient>
          <linearGradient id={`${logoId}-hood`} x1="32" x2="128" y1="34" y2="66" gradientUnits="userSpaceOnUse">
            <stop offset="0" stopColor="#fbfdff" />
            <stop offset="0.45" stopColor="#d8e1ec" />
            <stop offset="1" stopColor="#98a6b5" />
          </linearGradient>
          <linearGradient id={`${logoId}-casing`} x1="18" x2="42" y1="62" y2="103" gradientUnits="userSpaceOnUse">
            <stop offset="0" stopColor="#dfe6ef" />
            <stop offset="1" stopColor="#a1acba" />
          </linearGradient>
          <linearGradient id={`${logoId}-faceplate`} x1="41" x2="119" y1="44" y2="104" gradientUnits="userSpaceOnUse">
            <stop offset="0" stopColor="#21344d" />
            <stop offset="0.54" stopColor="#101c2b" />
            <stop offset="1" stopColor="#09121d" />
          </linearGradient>
          <linearGradient id={`${logoId}-muzzle`} x1="54" x2="106" y1="86" y2="123" gradientUnits="userSpaceOnUse">
            <stop offset="0" stopColor="#ffffff" />
            <stop offset="0.58" stopColor="#edf3fa" />
            <stop offset="1" stopColor="#d4dceb" />
          </linearGradient>
          <linearGradient id={`${logoId}-collar`} x1="52" x2="108" y1="118" y2="140" gradientUnits="userSpaceOnUse">
            <stop offset="0" stopColor="#7c3aed" />
            <stop offset="0.5" stopColor="#a855f7" />
            <stop offset="1" stopColor="#6d28d9" />
          </linearGradient>
          <radialGradient id={`${logoId}-iris`} cx="48%" cy="40%" r="66%">
            <stop offset="0" stopColor="#d4ffff" />
            <stop offset="0.42" stopColor="#71f5f1" />
            <stop offset="1" stopColor="#20c6d6" />
          </radialGradient>
          <radialGradient id={`${logoId}-cheek`} cx="50%" cy="50%" r="60%">
            <stop offset="0" stopColor="#ffffff" stopOpacity="0.3" />
            <stop offset="1" stopColor="#ffffff" stopOpacity="0" />
          </radialGradient>
          <clipPath id={`${logoId}-left-eye`}>
            <ellipse cx="57" cy="72" rx="14.5" ry="15" />
          </clipPath>
          <clipPath id={`${logoId}-right-eye`}>
            <ellipse cx="103" cy="72" rx="14.5" ry="15" />
          </clipPath>
        </defs>

        <ellipse cx="80" cy="86" rx="60" ry="44" fill="#08111e" opacity="0.13" />

        <path
          d="M45 27c-18 2-30 14-32 30-1 13 3 26 11 37 5 7 10 13 17 18 4-11 7-23 8-35 2-18-1-35-9-50Z"
          fill={`url(#${logoId}-ear)`}
          stroke="#232939"
          strokeWidth="3"
          strokeLinejoin="round"
        />
        <path d="M42 38c-10 2-17 11-18 22-1 10 3 19 10 29 3-8 5-18 5-28 1-8 0-16-3-23Z" fill="#202637" opacity="0.55" />
        <path
          d="M115 27c18 2 30 14 32 30 1 13-3 26-11 37-5 7-10 13-17 18-4-11-7-23-8-35-2-18 1-35 9-50Z"
          fill={`url(#${logoId}-ear)`}
          stroke="#232939"
          strokeWidth="3"
          strokeLinejoin="round"
        />
        <path d="M118 38c10 2 17 11 18 22 1 10-3 19-10 29-3-8-5-18-5-28-1-8 0-16 3-23Z" fill="#202637" opacity="0.55" />

        <path
          d="M31 49c1-16 14-29 31-29h36c17 0 30 13 31 29l8 5c6 4 8 10 8 17v14c0 21-16 39-38 43l-12 2H65l-12-2C31 124 15 106 15 85V71c0-7 2-13 8-17l8-5Z"
          fill={`url(#${logoId}-shell)`}
          stroke="#313a4d"
          strokeWidth="3.2"
          strokeLinejoin="round"
        />
        <path
          d="M38 35h84c10 0 18 7 21 18l2 7c-9-5-20-7-32-7H47c-12 0-23 2-32 7l2-7c3-11 11-18 21-18Z"
          fill={`url(#${logoId}-hood)`}
          stroke="#556274"
          strokeWidth="2.8"
          strokeLinejoin="round"
        />
        <path d="M49 39h63c11 0 19 6 23 16-8-4-18-6-29-6H54c-11 0-21 2-29 6 4-10 12-16 24-16Z" fill="#ffffff" fillOpacity="0.24" />
        <path d="M58 22c13-3 31-3 44 0 9 2 16 8 20 16" fill="none" stroke="#ffffff" strokeOpacity="0.72" strokeWidth="4" strokeLinecap="round" />

        <path
          d="M23 62h10c4 0 8 3 8 8v18c0 5-4 9-9 9h-8c-3 0-5-2-5-5V68c0-3 2-6 4-6Z"
          fill={`url(#${logoId}-casing)`}
          stroke="#667386"
          strokeWidth="2.2"
        />
        <path
          d="M127 62h10c2 0 4 3 4 6v24c0 3-2 5-5 5h-8c-5 0-9-4-9-9V70c0-5 4-8 8-8Z"
          fill={`url(#${logoId}-casing)`}
          stroke="#667386"
          strokeWidth="2.2"
        />
        <circle cx="28" cy="73" r="2.2" fill="#727f8f" />
        <circle cx="28" cy="87" r="2.2" fill="#727f8f" />
        <circle cx="132" cy="73" r="2.2" fill="#727f8f" />
        <circle cx="132" cy="87" r="2.2" fill="#727f8f" />

        <circle cx="80" cy="40" r="7.8" fill="#1e2b3f" stroke="#687489" strokeWidth="2.5" />
        <circle cx="80" cy="40" r="3.2" fill="#73f5ff" fillOpacity="0.55" />
        <circle cx="77.6" cy="37.8" r="1.2" fill="#efffff" fillOpacity="0.92" />
        <circle cx="64.5" cy="41.5" r="3.2" fill="#6f7c8c" />
        <circle cx="95.5" cy="41.5" r="3.2" fill="#6f7c8c" />
        <circle cx="48" cy="60" r="2.3" fill="#758394" />
        <circle cx="112" cy="60" r="2.3" fill="#758394" />

        <path
          d="M42 56c0-13 11-24 24-24h28c13 0 24 11 24 24v12c0 20-17 37-38 37s-38-17-38-37V56Z"
          fill={`url(#${logoId}-faceplate)`}
          stroke="#3a4758"
          strokeWidth="3"
        />
        <path d="M55 39h38c14 0 25 8 30 20-8-5-19-8-33-8H55Z" fill="#ffffff" fillOpacity="0.16" />
        <path d="M46 58c4-5 9-8 16-9" fill="none" stroke="#eef6ff" strokeOpacity="0.24" strokeWidth="3" strokeLinecap="round" />
        <path d="M114 58c-4-5-9-8-16-9" fill="none" stroke="#eef6ff" strokeOpacity="0.24" strokeWidth="3" strokeLinecap="round" />

        <g transform={`translate(0 ${EYE_CENTER_Y})`}>
          <g transform={`scale(1 ${eyeScaleY})`}>
            <g transform={`translate(0 ${-EYE_CENTER_Y})`}>
              <g clipPath={`url(#${logoId}-left-eye)`}>
                <ellipse cx="57" cy="72" rx="14.5" ry="15" fill="#edf8ff" />
                <ellipse cx="57" cy="73" rx="10.4" ry="11.2" fill={`url(#${logoId}-iris)`} />
                <g transform={`translate(${pupilOffset.x} ${pupilOffset.y})`}>
                  <circle cx="57" cy="73" r="4.15" fill="#06111d" />
                  <circle cx="58.8" cy="70.9" r="1.35" fill="#f9ffff" fillOpacity="0.98" />
                </g>
              </g>
              <g clipPath={`url(#${logoId}-right-eye)`}>
                <ellipse cx="103" cy="72" rx="14.5" ry="15" fill="#edf8ff" />
                <ellipse cx="103" cy="73" rx="10.4" ry="11.2" fill={`url(#${logoId}-iris)`} />
                <g transform={`translate(${pupilOffset.x} ${pupilOffset.y})`}>
                  <circle cx="103" cy="73" r="4.15" fill="#06111d" />
                  <circle cx="104.8" cy="70.9" r="1.35" fill="#f9ffff" fillOpacity="0.98" />
                </g>
              </g>
            </g>
          </g>
        </g>

        <path d="M46 64c4-6 10-10 18-10" fill="none" stroke="#0e1926" strokeOpacity="0.42" strokeWidth="3.2" strokeLinecap="round" />
        <path d="M114 64c-4-6-10-10-18-10" fill="none" stroke="#0e1926" strokeOpacity="0.42" strokeWidth="3.2" strokeLinecap="round" />
        <ellipse cx="54" cy="95" rx="11" ry="8.5" fill={`url(#${logoId}-cheek)`} />
        <ellipse cx="106" cy="95" rx="11" ry="8.5" fill={`url(#${logoId}-cheek)`} />

        <path
          d="M50 90c4-10 16-17 30-17s26 7 30 17c5 14-5 29-22 32-5 1-11 1-16 0-17-3-27-18-22-32Z"
          fill={`url(#${logoId}-muzzle)`}
          stroke="#5f6b80"
          strokeWidth="2.5"
          strokeLinejoin="round"
        />
        <path d="M61 84c7-5 31-5 38 0" fill="none" stroke="#ffffff" strokeOpacity="0.46" strokeWidth="3" strokeLinecap="round" />
        <ellipse cx="80" cy="97" rx="8.6" ry="6.7" fill="#273243" />
        <ellipse cx="80" cy="99" rx="4.1" ry="2.5" fill="#18212d" />
        <circle cx="77.4" cy="94.7" r="1.35" fill="#edffff" fillOpacity="0.9" />
        <path d="M80 104v5.4" fill="none" stroke="#273243" strokeWidth="2.5" strokeLinecap="round" />
        <path
          d="M66.5 110.5c3 4.6 7.9 7 13.5 7s10.5-2.4 13.5-7"
          fill="none"
          stroke="#283345"
          strokeWidth="3.4"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path d="M60.5 103.8c2.4 3.3 5.8 5 10.2 5" fill="none" stroke="#c8d2df" strokeWidth="2.1" strokeLinecap="round" />
        <path d="M99.5 103.8c-2.4 3.3-5.8 5-10.2 5" fill="none" stroke="#c8d2df" strokeWidth="2.1" strokeLinecap="round" />

        <path d="M53 121h54c4 0 7 3 7 7v7c0 4-3 7-7 7H53c-4 0-7-3-7-7v-7c0-4 3-7 7-7Z" fill={`url(#${logoId}-collar)`} />
        <text
          x="80"
          y="133"
          textAnchor="middle"
          fontSize="9"
          fontWeight="800"
          letterSpacing="1.5"
          fill="#f8e8ff"
          fillOpacity={collarTextOpacity}
        >
          BANTAY
        </text>
        <line x1="80" y1="142" x2="80" y2="149" stroke="#aab2c0" strokeWidth="3" strokeLinecap="round" />
        <circle cx="80" cy="152" r="9" fill="#7c3aed" stroke="#43386e" strokeWidth="2.2" />
        <circle cx="80" cy="152" r="4.8" fill="#62fff0" fillOpacity="0.24" />
        <text x="80" y="155.5" textAnchor="middle" fontSize="10" fontWeight="900" fill="#71fff1">B</text>
      </svg>

      <div
        className={cn(
          "pointer-events-none absolute inset-x-2 top-1/2 h-10 -translate-y-1/2 rounded-full bg-cyan-300/10 blur-xl transition-opacity duration-300",
          isHovered ? "opacity-100" : "opacity-70",
          prefersReducedMotion && "opacity-50",
        )}
      />

      <style jsx>{`
        .bantay-logo-idle {
          animation: bantayFloat 5.8s ease-in-out infinite;
        }

        @keyframes bantayFloat {
          0%,
          100% {
            transform: translateY(0px);
          }
          50% {
            transform: translateY(-1.5px);
          }
        }
      `}</style>
    </div>
  )
}