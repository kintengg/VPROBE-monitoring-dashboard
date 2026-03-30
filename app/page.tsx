"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import { usePathname, useRouter, useSearchParams } from "next/navigation"
import { VideoGrid } from "@/components/surveillance/video-grid"
import { EventFeed } from "@/components/surveillance/event-feed"
import { AISearchBar } from "@/components/surveillance/ai-search-bar"
import { AddLocationModal } from "@/components/surveillance/add-location-modal"
import { AddVideoModal } from "@/components/surveillance/add-video-modal"
import { LocationMap } from "@/components/surveillance/location-map"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { useUploadQueue } from "@/components/uploads/upload-queue-provider"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Switch } from "@/components/ui/switch"
import { FootageDatePicker } from "@/components/ui/footage-date-picker"
import { AlertCircle, Calendar, ChevronDown, Loader2, MapPin, Pencil, Plus, ScanLine, Trash2, Video } from "lucide-react"
import {
  createLocation,
  deleteLocation,
  getEvents,
  getLocations,
  type EventRecord,
  type LocationRecord,
  updateLocation,
} from "@/lib/api"

export default function SurveillancePage() {
  const { enqueueUploads, settledUploadsVersion } = useUploadQueue()
  const router = useRouter()
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const [detectionMode, setDetectionMode] = useState(true)
  const [locationMenuOpen, setLocationMenuOpen] = useState(false)
  const [locationModalOpen, setLocationModalOpen] = useState(false)
  const [editingLocation, setEditingLocation] = useState<LocationRecord | null>(null)
  const [pendingDeleteLocation, setPendingDeleteLocation] = useState<LocationRecord | null>(null)
  const [isDeletingLocation, setIsDeletingLocation] = useState(false)
  const [videoModalOpen, setVideoModalOpen] = useState(false)
  const [selectedUploadLocationId, setSelectedUploadLocationId] = useState("")
  const [selectedDate, setSelectedDate] = useState("")
  const [locations, setLocations] = useState<LocationRecord[]>([])
  const [events, setEvents] = useState<EventRecord[]>([])
  const [locationsLoading, setLocationsLoading] = useState(true)
  const [eventsLoading, setEventsLoading] = useState(true)
  const [pageError, setPageError] = useState<string | null>(null)
  const [locationsError, setLocationsError] = useState<string | null>(null)
  const [eventsError, setEventsError] = useState<string | null>(null)

  const loadLocations = useCallback(async () => {
    setLocationsLoading(true)
    try {
      const response = await getLocations()
      setLocations(response)
      setLocationsError(null)
    } catch (error) {
      setLocationsError(error instanceof Error ? error.message : "Failed to load locations.")
    } finally {
      setLocationsLoading(false)
    }
  }, [])

  const loadEvents = useCallback(async () => {
    setEventsLoading(true)
    try {
      const response = await getEvents()
      setEvents(response)
      setEventsError(null)
    } catch (error) {
      setEventsError(error instanceof Error ? error.message : "Failed to load events.")
    } finally {
      setEventsLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadLocations()
  }, [loadLocations])

  useEffect(() => {
    void loadEvents()
  }, [loadEvents])

  useEffect(() => {
    if (settledUploadsVersion === 0) {
      return
    }

    void Promise.all([loadLocations(), loadEvents()])
  }, [loadEvents, loadLocations, settledUploadsVersion])

  const filteredLocations = useMemo(() => {
    if (!selectedDate) return locations

    return locations.map((location) => ({
      ...location,
      videos: location.videos.filter((video) => video.date === selectedDate),
    }))
  }, [locations, selectedDate])

  const hasAnyVideosForSelectedDate = useMemo(
    () => filteredLocations.some((location) => location.videos.length > 0),
    [filteredLocations],
  )
  const footageDates = useMemo(
    () => Array.from(new Set(locations.flatMap((location) => location.videos.map((video) => video.date)))).sort(),
    [locations],
  )

  const handleVideoModalChange = (open: boolean) => {
    setVideoModalOpen(open)
    if (!open) {
      setSelectedUploadLocationId("")
    }
  }

  const handleLocationModalChange = (open: boolean) => {
    setLocationModalOpen(open)
    if (!open) {
      setEditingLocation(null)
    }
  }

  const handleOpenAddVideo = useCallback((locationId?: string) => {
    setSelectedUploadLocationId(locationId ?? "")
    setVideoModalOpen(true)
  }, [])

  useEffect(() => {
    if (searchParams.get("openAddVideo") !== "1") {
      return
    }

    handleOpenAddVideo()

    const nextParams = new URLSearchParams(searchParams.toString())
    nextParams.delete("openAddVideo")
    const nextQuery = nextParams.toString()
    router.replace(nextQuery ? `${pathname}?${nextQuery}` : pathname, { scroll: false })
  }, [handleOpenAddVideo, pathname, router, searchParams])

  const handleOpenAddLocation = () => {
    setLocationMenuOpen(false)
    setTimeout(() => {
      setEditingLocation(null)
      setLocationModalOpen(true)
    }, 0)
  }

  const handleOpenEditLocation = (location: LocationRecord) => {
    setLocationMenuOpen(false)
    setTimeout(() => {
      setEditingLocation(location)
      setLocationModalOpen(true)
    }, 0)
  }

  const handleOpenDeleteLocation = (location: LocationRecord) => {
    setLocationMenuOpen(false)
    setTimeout(() => {
      setPendingDeleteLocation(location)
    }, 0)
  }

  const handleSaveLocation = async (data: {
    name: string
    latitude: number
    longitude: number
    description: string
    address: string
  }) => {
    try {
      setPageError(null)
      if (editingLocation) {
        await updateLocation(editingLocation.id, data)
      } else {
        await createLocation(data)
      }
      await Promise.all([loadLocations(), loadEvents()])
    } catch (error) {
      const message = error instanceof Error ? error.message : editingLocation ? "Failed to update location." : "Failed to create location."
      setPageError(message)
      throw error
    }
  }

  const handleDeleteSelectedLocation = async () => {
    if (!pendingDeleteLocation || isDeletingLocation) return

    try {
      setIsDeletingLocation(true)
      setPageError(null)
      await deleteLocation(pendingDeleteLocation.id)
      if (selectedUploadLocationId === pendingDeleteLocation.id) {
        setSelectedUploadLocationId("")
      }
      if (editingLocation?.id === pendingDeleteLocation.id) {
        setEditingLocation(null)
        setLocationModalOpen(false)
      }
      setPendingDeleteLocation(null)
      await Promise.all([loadLocations(), loadEvents()])
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to delete location."
      setPageError(message)
    } finally {
      setIsDeletingLocation(false)
    }
  }

  const handleAddVideo = async (upload: {
    file: File
    locationId: string
    date: string
    startTime: string
    endTime: string
    fastMode: boolean
  }) => {
    try {
      setPageError(null)
      enqueueUploads([
        {
          ...upload,
          locationName: locations.find((location) => location.id === upload.locationId)?.name ?? "Unknown location",
        },
      ])
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to queue video upload."
      setPageError(message)
      throw error
    }
  }

  const activeError = pageError ?? locationsError ?? eventsError

  return (
    <div className="flex h-full">
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="flex items-center justify-between px-6 py-4 border-b border-border bg-card/50 backdrop-blur-sm">
          <h1 className="text-xl font-semibold text-white shrink-0 mr-6">Surveillance Overview</h1>
          
          <div className="flex items-center gap-3 flex-wrap justify-end">
            {/* Date Filter */}
            <FootageDatePicker
              value={selectedDate}
              onChange={setSelectedDate}
              highlightedDates={footageDates}
              allowClear
            />

            {/* Detection Mode Toggle */}
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-2xl bg-secondary border border-border">
              <ScanLine className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-foreground">Detection</span>
              <Switch
                checked={detectionMode}
                onCheckedChange={setDetectionMode}
                className="data-[state=checked]:bg-accent"
              />
            </div>

            <DropdownMenu open={locationMenuOpen} onOpenChange={setLocationMenuOpen}>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="border-border text-foreground hover:bg-secondary rounded-2xl px-4">
                  <MapPin className="w-4 h-4 mr-2" />
                  Location
                  <ChevronDown className="ml-2 h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuItem onSelect={handleOpenAddLocation}>
                  <Plus className="h-4 w-4" />
                  Add Location
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger disabled={locations.length === 0}>
                    <Pencil className="h-4 w-4" />
                    Edit Location
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent>
                    {locations.length > 0 ? (
                      locations.map((location) => (
                        <DropdownMenuItem key={`edit-${location.id}`} onSelect={() => handleOpenEditLocation(location)}>
                          {location.name}
                        </DropdownMenuItem>
                      ))
                    ) : (
                      <DropdownMenuItem disabled>No locations available</DropdownMenuItem>
                    )}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
                <DropdownMenuSeparator />
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger disabled={locations.length === 0}>
                    <Trash2 className="h-4 w-4" />
                    Delete Location
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent>
                    {locations.length > 0 ? (
                      locations.map((location) => (
                        <DropdownMenuItem
                          key={`delete-${location.id}`}
                          variant="destructive"
                          onSelect={() => handleOpenDeleteLocation(location)}
                        >
                          {location.name}
                        </DropdownMenuItem>
                      ))
                    ) : (
                      <DropdownMenuItem disabled>No locations available</DropdownMenuItem>
                    )}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Add Video Button */}
            <Button
              className="bg-primary text-primary-foreground hover:bg-primary/90 rounded-2xl px-4 shadow-elevated-sm"
              onClick={() => handleOpenAddVideo()}
            >
              <Video className="w-4 h-4 mr-2" />
              Add Video
            </Button>
          </div>
        </header>

        {/* Video Grid */}
        <div className="flex-1 overflow-auto p-6">
          {activeError && (
            <div className="mb-4 flex items-start gap-3 rounded-2xl border border-destructive/30 bg-destructive/10 p-4 text-sm text-destructive">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{activeError}</span>
            </div>
          )}

          {selectedDate && !hasAnyVideosForSelectedDate && filteredLocations.length > 0 && (
            <div className="mb-4 flex items-start gap-3 rounded-2xl border border-primary/20 bg-primary/5 p-4 text-sm text-foreground">
              <Calendar className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
              <div>
                <p className="font-medium">No footage is scheduled for this date yet.</p>
                <p className="mt-1 text-muted-foreground">You can still use any placeholder card below to upload a new recording directly into the correct location.</p>
              </div>
            </div>
          )}

          {locationsLoading ? (
            <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground">
              <Loader2 className="mb-4 h-8 w-8 animate-spin" />
              <p className="text-sm">Loading locations and videos...</p>
            </div>
          ) : filteredLocations.length > 0 ? (
            <VideoGrid
              locations={filteredLocations}
              detectionMode={detectionMode}
              onAddVideoClick={handleOpenAddVideo}
            />
          ) : (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="w-16 h-16 rounded-3xl bg-secondary flex items-center justify-center mb-4">
                <Video className="w-8 h-8 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-medium text-foreground mb-2">No videos found</h3>
              <p className="text-sm text-muted-foreground mb-4">
                {selectedDate 
                  ? `No videos available for ${new Date(selectedDate).toLocaleDateString()}`
                  : "Add a video to get started"
                }
              </p>
              {selectedDate && (
                <Button variant="outline" onClick={() => setSelectedDate("")} className="rounded-2xl">
                  Clear date filter
                </Button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Right Sidebar */}
      <aside className="w-80 border-l border-border bg-card/30 flex flex-col h-full">
        <AISearchBar />
        {/* Location Map - Below Search Bar */}
        <div className="px-4 pb-4">
          <LocationMap locations={filteredLocations} />
        </div>
        <EventFeed events={events} loading={eventsLoading} />
      </aside>

      {/* Modals */}
      <AddLocationModal 
        open={locationModalOpen} 
        onOpenChange={handleLocationModalChange}
        initialData={editingLocation}
        onSubmitLocation={handleSaveLocation}
      />
      <AddVideoModal 
        open={videoModalOpen}
        onOpenChange={handleVideoModalChange}
        locations={locations.map((location) => ({ id: location.id, name: location.name }))}
        initialLocationId={selectedUploadLocationId}
        onAddVideo={handleAddVideo}
      />
      <AlertDialog open={Boolean(pendingDeleteLocation)} onOpenChange={(open) => !open && setPendingDeleteLocation(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {pendingDeleteLocation?.name}?</AlertDialogTitle>
            <AlertDialogDescription>
              This will remove the location, all linked footage, generated processed videos, and related detection events.
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeletingLocation}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={isDeletingLocation}
              onClick={(event) => {
                event.preventDefault()
                void handleDeleteSelectedLocation()
              }}
            >
              {isDeletingLocation ? "Deleting..." : "Delete Location"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
