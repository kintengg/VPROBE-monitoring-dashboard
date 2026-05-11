"use client"

import { Calendar, Car, Clock, Gauge, MapPin, Users } from "lucide-react"

interface VideoMetadataProps {
  date: string
  startTime: string
  endTime: string
  gpsLat: number
  gpsLng: number
  trackedPedestriansSoFar: number
  pedestrianCount: number
  isVehicle?: boolean
  currentLOS?: string | null
  worstLOS?: string | null
  averageLOS?: string | null
}

export function VideoMetadata({
  date,
  startTime,
  endTime,
  gpsLat,
  gpsLng,
  trackedPedestriansSoFar,
  pedestrianCount,
  isVehicle = false,
  currentLOS,
  worstLOS,
  averageLOS,
}: VideoMetadataProps) {
  const subjectIcon = isVehicle ? Car : Users
  const totalLabel = isVehicle ? "Total Vehicles" : "Total Pedestrians"
  const trackedLabel = isVehicle ? "Tracked So Far" : "Tracked So Far"

  return (
    <div className={`grid gap-4 grid-cols-2 ${isVehicle ? "md:grid-cols-4 xl:grid-cols-8" : "md:grid-cols-3 xl:grid-cols-5"}`}>
      <MetadataCard icon={Calendar} label="Date" value={date} />
      <MetadataCard icon={Clock} label="Time Range" value={`${startTime} - ${endTime}`} />
      <MetadataCard icon={MapPin} label="GPS Location" value={`${gpsLat.toFixed(4)}, ${gpsLng.toFixed(4)}`} />
      <MetadataCard icon={subjectIcon} label={trackedLabel} value={trackedPedestriansSoFar.toString()} highlight />
      <MetadataCard icon={subjectIcon} label={totalLabel} value={pedestrianCount.toString()} highlight />
      {isVehicle && (
        <>
          <MetadataCard icon={Gauge} label="Current LOS" value={currentLOS ?? "--"} highlight />
          <MetadataCard icon={Gauge} label="Worst LOS" value={worstLOS ?? "--"} highlight />
          <MetadataCard icon={Gauge} label="Avg LOS" value={averageLOS ?? "--"} highlight />
        </>
      )}
    </div>
  )
}

function MetadataCard({
  icon: Icon,
  label,
  value,
  highlight = false,
}: {
  icon: React.ElementType
  label: string
  value: string
  highlight?: boolean
}) {
  return (
    <div className="p-4 rounded-lg bg-card border border-border">
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`w-4 h-4 ${highlight ? "text-primary" : "text-muted-foreground"}`} />
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <p className={`text-sm font-medium ${highlight ? "text-primary" : "text-foreground"}`}>
        {value}
      </p>
    </div>
  )
}
