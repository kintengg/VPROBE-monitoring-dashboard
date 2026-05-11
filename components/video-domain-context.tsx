"use client"

import { createContext, useContext, useEffect, useState } from "react"

export type VideoDomain = "pedestrian" | "vehicle" | null

interface VideoDomainContextValue {
  videoDomain: VideoDomain
  setVideoDomain: (domain: VideoDomain) => void
}

const VideoDomainContext = createContext<VideoDomainContextValue | null>(null)

export function VideoDomainProvider({ children }: { children: React.ReactNode }) {
  const [videoDomain, setVideoDomain] = useState<VideoDomain>(null)
  return (
    <VideoDomainContext.Provider value={{ videoDomain, setVideoDomain }}>
      {children}
    </VideoDomainContext.Provider>
  )
}

export function useVideoDomain() {
  const ctx = useContext(VideoDomainContext)
  return ctx?.videoDomain ?? null
}

export function useSetVideoDomain() {
  const ctx = useContext(VideoDomainContext)
  useEffect(() => {
    return () => ctx?.setVideoDomain(null)
  }, [ctx])
  return ctx?.setVideoDomain ?? ((_: VideoDomain) => undefined)
}
