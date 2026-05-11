import type { VehicleLOS } from "@/lib/api"

export const LOS_HEX: Record<VehicleLOS, string> = {
  A: "#10b981", // emerald-500
  B: "#84cc16", // lime-500
  C: "#eab308", // yellow-500
  D: "#f97316", // orange-500
  E: "#f43f5e", // rose-500
  F: "#dc2626", // red-600
}

export const LOS_UNKNOWN_HEX = "#64748b" // slate-500

export const LOS_PILL_CLASS: Record<VehicleLOS, string> = {
  A: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  B: "bg-lime-500/15 text-lime-300 border-lime-500/30",
  C: "bg-yellow-500/15 text-yellow-300 border-yellow-500/30",
  D: "bg-orange-500/15 text-orange-300 border-orange-500/30",
  E: "bg-rose-500/15 text-rose-300 border-rose-500/30",
  F: "bg-red-600/20 text-red-300 border-red-600/40",
}

export const LOS_PILL_NEUTRAL = "bg-secondary/40 text-muted-foreground border-border/60"

export const LOS_GRADES: ReadonlyArray<VehicleLOS> = ["A", "B", "C", "D", "E", "F"]
