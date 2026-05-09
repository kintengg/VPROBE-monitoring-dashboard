"use client"

import { Calendar, Car, Clock, MapPin, Users } from "lucide-react"

interface VideoMetadataProps {
  date: string
  startTime: string
  endTime: string
  gpsLat: number
  gpsLng: number
  trackedPedestriansSoFar: number
  pedestrianCount: number
  isVehicle?: boolean
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
}: VideoMetadataProps) {
  const subjectIcon = isVehicle ? Car : Users
  const totalLabel = isVehicle ? "Total Vehicles" : "Total Pedestrians"
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-5">
      <MetadataCard
        icon={Calendar}
        label="Date"
        value={date}
      />
      <MetadataCard
        icon={Clock}
        label="Time Range"
        value={`${startTime} - ${endTime}`}
      />
      <MetadataCard
        icon={MapPin}
        label="GPS Location"
        value={`${gpsLat.toFixed(4)}, ${gpsLng.toFixed(4)}`}
      />
      <MetadataCard
        icon={subjectIcon}
        label="Tracked So Far"
        value={trackedPedestriansSoFar.toString()}
        highlight
      />
      <MetadataCard
        icon={subjectIcon}
        label={totalLabel}
        value={pedestrianCount.toString()}
        highlight
      />
    </div>
  )
}

function MetadataCard({ 
  icon: Icon, 
  label, 
  value, 
  highlight = false 
}: { 
  icon: React.ElementType
  label: string
  value: string
  highlight?: boolean
}) {
  return (
    <div className="p-4 rounded-lg bg-card border border-border">
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`w-4 h-4 ${highlight ? 'text-primary' : 'text-muted-foreground'}`} />
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <p className={`text-sm font-medium ${highlight ? 'text-primary' : 'text-foreground'}`}>
        {value}
      </p>
    </div>
  )
}
