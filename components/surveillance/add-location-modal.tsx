"use client"

import { useState, useEffect, useCallback } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card"
import { Check, Copy, Info, Loader2, MapPin, Search } from "lucide-react"
import {
  parseEntryExitPointsConfiguration,
  searchLocations,
  type GateDirectionConfiguration,
  type LocationPayload,
  type ROIConfiguration,
} from "@/lib/api"

const ROI_PROMPT_TEMPLATE = `I am going to paste CVAT polygon coordinates for one camera/location. Please convert them into the exact JSON format required by my application.

Rules:
1. Normalize x by image width.
2. Normalize y by image height.
3. Keep 6 decimal places.
4. Output valid JSON only.
5. Do not add explanations, markdown, comments, or extra text.
6. Treat all polygons I provide as walkable include polygons unless I explicitly say otherwise.
7. The output format must be exactly:

{
  "referenceSize": [WIDTH, HEIGHT],
  "includePolygonsNorm": [
    [[x1, y1], [x2, y2], [x3, y3]],
    [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  ]
}

Image size:
WIDTH = 1920
HEIGHT = 1080

Here are the raw CVAT coordinates:
[PASTE CVAT POLYGON COORDINATES HERE]`

const ENTRY_EXIT_PROMPT_TEMPLATE = `I am going to provide:
1. the raw image
2. the same image with my directional strip annotations
3. the annotation file
4. a short description of which direction should count as entering and which should count as exiting

Please convert my directional strip annotations into the exact JSON format required by my application.

Rules:
1. Normalize x by image width.
2. Normalize y by image height.
3. Keep 6 decimal places.
4. Output valid JSON only.
5. Do not add explanations, markdown, comments, or extra text.
6. The three strips must be assigned in sequential order across the pedestrian path:
   - strip_0
   - strip_1
   - strip_2
7. Use the direction description I provide to decide:
   - path_0_1_2 = entering or exiting
   - path_2_1_0 = the opposite
8. Output format must be exactly:

{
  "referenceSize": [WIDTH, HEIGHT],
  "gateDirectionZonesNorm": {
    "strip_0": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "strip_1": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
    "strip_2": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
  },
  "directionMapping": {
    "path_0_1_2": "entering",
    "path_2_1_0": "exiting"
  }
}`

const SEARCHABLE_LOCATIONS: Array<{
  aliases: string[]
  lat: number
  lng: number
  address: string
}> = [
  {
    aliases: ["edsa sec walk", "edsa sec walkway", "xavier hall", "xavier"],
    lat: 14.6397,
    lng: 121.0775,
    address: "EDSA Sec Walk, Xavier Hall, Ateneo de Manila University",
  },
  {
    aliases: ["kostka walk", "kostka walkway", "kostka hall", "kostka"],
    lat: 14.639,
    lng: 121.0781,
    address: "Kostka Walk, Kostka Hall, Ateneo de Manila University",
  },
  {
    aliases: ["gate 1", "gate 1 walkway", "gate one"],
    lat: 14.6418,
    lng: 121.0758,
    address: "Gate 1 Walkway, Ateneo de Manila University",
  },
  {
    aliases: ["gate 3", "gate 3 walkway", "gate three"],
    lat: 14.6376,
    lng: 121.0742,
    address: "Gate 3 Walkway, Ateneo de Manila University",
  },
  {
    aliases: ["manila"],
    lat: 14.5995,
    lng: 120.9842,
    address: "Manila, Philippines",
  },
  {
    aliases: ["tokyo"],
    lat: 35.6762,
    lng: 139.6503,
    address: "Tokyo, Japan",
  },
]

function normalizeSearchQuery(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim()
}

function findFallbackMatch(normalizedQuery: string) {
  return SEARCHABLE_LOCATIONS.find(({ aliases }) => aliases.some((alias) => normalizedQuery.includes(alias) || alias.includes(normalizedQuery)))
}

function parseROIConfiguration(value: string): ROIConfiguration | null {
  const trimmed = value.trim()
  if (!trimmed) {
    return null
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(trimmed)
  } catch {
    throw new Error("ROI JSON must be valid JSON.")
  }

  if (!parsed || typeof parsed !== "object") {
    throw new Error("ROI JSON must be an object.")
  }

  const record = parsed as Record<string, unknown>
  const referenceSize = record.referenceSize
  const includePolygonsNorm = record.includePolygonsNorm

  if (
    !Array.isArray(referenceSize) ||
    referenceSize.length !== 2 ||
    referenceSize.some((entry) => typeof entry !== "number" || !Number.isFinite(entry) || entry <= 0)
  ) {
    throw new Error("ROI JSON must include a numeric referenceSize like [1920, 1080].")
  }

  if (!Array.isArray(includePolygonsNorm) || includePolygonsNorm.length === 0) {
    throw new Error("ROI JSON must include at least one polygon in includePolygonsNorm.")
  }

  const polygons = includePolygonsNorm.map((polygon) => {
    if (!Array.isArray(polygon) || polygon.length < 3) {
      throw new Error("Each ROI polygon must contain at least 3 normalized points.")
    }

    return polygon.map((point) => {
      if (!Array.isArray(point) || point.length !== 2) {
        throw new Error("Each ROI point must be a two-number array like [0.25, 0.98].")
      }

      const [x, y] = point
      if (
        typeof x !== "number" ||
        typeof y !== "number" ||
        !Number.isFinite(x) ||
        !Number.isFinite(y) ||
        x < 0 ||
        x > 1 ||
        y < 0 ||
        y > 1
      ) {
        throw new Error("ROI points must use normalized coordinates between 0 and 1.")
      }

      return [x, y] as [number, number]
    })
  })

  return {
    referenceSize: [referenceSize[0] as number, referenceSize[1] as number],
    includePolygonsNorm: polygons,
  }
}

interface AddLocationModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  initialData?: LocationPayload | null
  onSubmitLocation?: (data: LocationPayload) => void | Promise<void>
}

export function AddLocationModal({ open, onOpenChange, initialData = null, onSubmitLocation }: AddLocationModalProps) {
  const [name, setName] = useState("")
  const [latitude, setLatitude] = useState("")
  const [longitude, setLongitude] = useState("")
  const [description, setDescription] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const [address, setAddress] = useState("")
  const [walkableAreaM2, setWalkableAreaM2] = useState("")
  const [roiCoordinatesText, setRoiCoordinatesText] = useState("")
  const [entryExitPointsText, setEntryExitPointsText] = useState("")
  const [searchError, setSearchError] = useState<string | null>(null)
  const [walkableAreaError, setWalkableAreaError] = useState<string | null>(null)
  const [roiError, setRoiError] = useState<string | null>(null)
  const [entryExitPointsError, setEntryExitPointsError] = useState<string | null>(null)
  const [isSearching, setIsSearching] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [roiPromptCopied, setRoiPromptCopied] = useState(false)
  const [entryExitPromptCopied, setEntryExitPromptCopied] = useState(false)
  const isEditing = Boolean(initialData)

  const validateWalkableArea = useCallback((value: string) => {
    if (!value.trim()) {
      return null
    }

    const parsed = Number.parseFloat(value)
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return "Walkable area must be a positive number in square meters."
    }

    return null
  }, [])

  const validateEntryExitPointsText = useCallback((value: string) => {
    if (!value.trim()) {
      return null
    }

    try {
      parseEntryExitPointsConfiguration(value)
      return null
    } catch (error) {
      return error instanceof Error ? error.message : "Entry/Exit Points JSON must be valid JSON."
    }
  }, [])

  const resetForm = useCallback((nextData?: LocationPayload | null) => {
    setName(nextData?.name ?? "")
    setLatitude(nextData ? nextData.latitude.toString() : "")
    setLongitude(nextData ? nextData.longitude.toString() : "")
    setDescription(nextData?.description ?? "")
    setSearchQuery(nextData?.address ?? nextData?.name ?? "")
    setAddress(nextData?.address ?? "")
    setWalkableAreaM2(nextData?.walkableAreaM2 != null ? nextData.walkableAreaM2.toString() : "")
    setRoiCoordinatesText(nextData?.roiCoordinates ? JSON.stringify(nextData.roiCoordinates, null, 2) : "")
    setEntryExitPointsText(nextData?.entryExitPoints ? JSON.stringify(nextData.entryExitPoints, null, 2) : "")
    setSearchError(null)
    setWalkableAreaError(null)
    setRoiError(null)
    setEntryExitPointsError(null)
    setRoiPromptCopied(false)
    setEntryExitPromptCopied(false)
  }, [])

  useEffect(() => {
    if (!open) return
    resetForm(initialData)
  }, [initialData, open, resetForm])

  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim() || isSearching) return

    setIsSearching(true)
    setSearchError(null)

    const normalizedQuery = normalizeSearchQuery(searchQuery)
    const fallbackMatch = findFallbackMatch(normalizedQuery)
    let foundResult = false
    let remoteSearchError: string | null = null

    try {
      const results = await searchLocations(searchQuery.trim())
      const firstResult = results[0]

      if (firstResult) {
        setLatitude(firstResult.latitude.toString())
        setLongitude(firstResult.longitude.toString())
        setAddress(firstResult.address)
        if (!name.trim()) {
          setName(firstResult.name || searchQuery.trim())
        }
        foundResult = true
      }
    } catch (error) {
      remoteSearchError = error instanceof Error ? error.message : "Location search is unavailable right now."
    } finally {
      if (!foundResult && fallbackMatch) {
        setLatitude(fallbackMatch.lat.toString())
        setLongitude(fallbackMatch.lng.toString())
        setAddress(fallbackMatch.address)
        if (!name.trim()) {
          setName(searchQuery.trim())
        }
        foundResult = true
      }

      if (!foundResult) {
        setSearchError(remoteSearchError ?? "Location not found. Try a more specific name or enter the coordinates manually.")
      }

      setIsSearching(false)
    }
  }, [isSearching, name, searchQuery])

  const handleWalkableAreaChange = useCallback((value: string) => {
    setWalkableAreaM2(value)
    setWalkableAreaError(validateWalkableArea(value))
  }, [validateWalkableArea])

  const handleEntryExitPointsChange = useCallback((value: string) => {
    setEntryExitPointsText(value)
    setEntryExitPointsError(validateEntryExitPointsText(value))
  }, [validateEntryExitPointsText])

  const handleCopyRoiPrompt = useCallback(async () => {
    if (typeof navigator === "undefined" || !navigator.clipboard) {
      return
    }

    await navigator.clipboard.writeText(ROI_PROMPT_TEMPLATE)
    setRoiPromptCopied(true)
    window.setTimeout(() => setRoiPromptCopied(false), 1800)
  }, [])

  const handleCopyEntryExitPrompt = useCallback(async () => {
    if (typeof navigator === "undefined" || !navigator.clipboard) {
      return
    }

    await navigator.clipboard.writeText(ENTRY_EXIT_PROMPT_TEMPLATE)
    setEntryExitPromptCopied(true)
    window.setTimeout(() => setEntryExitPromptCopied(false), 1800)
  }, [])

  const handleSubmit = async () => {
    if (!name || !latitude || !longitude || isSubmitting) return

    let parsedROI: ROIConfiguration | null = null
    let parsedEntryExitPoints: GateDirectionConfiguration | null = null
    let parsedWalkableArea: number | null = null

    try {
      parsedROI = parseROIConfiguration(roiCoordinatesText)
      setRoiError(null)
    } catch (error) {
      setRoiError(error instanceof Error ? error.message : "Invalid ROI configuration.")
      return
    }

    try {
      parsedEntryExitPoints = parseEntryExitPointsConfiguration(entryExitPointsText)
      setEntryExitPointsError(null)
    } catch (error) {
      setEntryExitPointsError(error instanceof Error ? error.message : "Entry/Exit Points JSON must be valid JSON.")
      return
    }

    const nextWalkableAreaError = validateWalkableArea(walkableAreaM2)
    if (nextWalkableAreaError) {
      setWalkableAreaError(nextWalkableAreaError)
      return
    }

    setWalkableAreaError(null)
    if (walkableAreaM2.trim()) {
      parsedWalkableArea = Number.parseFloat(walkableAreaM2)
    }

    setIsSubmitting(true)
    try {
      await onSubmitLocation?.({
        name,
        latitude: parseFloat(latitude),
        longitude: parseFloat(longitude),
        description,
        address,
        roiCoordinates: parsedROI,
        entryExitPoints: parsedEntryExitPoints,
        walkableAreaM2: parsedWalkableArea,
      })
      handleClose()
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleClose = () => {
    resetForm(null)
    setIsSearching(false)
    setSearchError(null)
    setRoiPromptCopied(false)
    setEntryExitPromptCopied(false)
    onOpenChange(false)
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (!nextOpen) {
          handleClose()
        } else {
          onOpenChange(nextOpen)
        }
      }}
    >
      <DialogContent className="border-border bg-card flex max-h-[90vh] flex-col gap-0 overflow-hidden p-0 sm:max-w-md sm:rounded-2xl">
        <DialogHeader className="shrink-0 px-6 pt-6">
          <DialogTitle className="flex items-center gap-2 text-foreground">
            <MapPin className="w-5 h-5" />
            {isEditing ? "Edit Location" : "Add New Location"}
          </DialogTitle>
          <DialogDescription className="text-muted-foreground">
            {isEditing
              ? "Update the location details, camera coordinates, ROI polygons, and walkable area."
              : "Search for a place or enter GPS coordinates, ROI polygons, and walkable area manually."}
          </DialogDescription>
        </DialogHeader>
        
        <div className="smooth-scrollbar min-h-0 space-y-4 overflow-y-auto overscroll-contain px-6 py-4">
          {/* Place Search */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Search Location</label>
            <div className="flex gap-2">
              <Input 
                placeholder="Search for a place..." 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault()
                    void handleSearch()
                  }
                }}
                className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
              />
              <Button 
                variant="outline" 
                onClick={() => void handleSearch()}
                disabled={isSearching || isSubmitting}
                className="border-border shrink-0"
              >
                {isSearching ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Search className="w-4 h-4" />
                )}
              </Button>
            </div>
            {address && (
              <p className="text-xs text-muted-foreground">Found: {address}</p>
            )}
            {searchError && (
              <p className="text-xs text-destructive">{searchError}</p>
            )}
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Location Name</label>
            <Input 
              placeholder="e.g., Gate 1 Walkway" 
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
            />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Latitude</label>
              <Input 
                type="number" 
                step="0.000001"
                placeholder="40.7128" 
                value={latitude}
                onChange={(e) => setLatitude(e.target.value)}
                className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Longitude</label>
              <Input 
                type="number" 
                step="0.000001"
                placeholder="-74.0060" 
                value={longitude}
                onChange={(e) => setLongitude(e.target.value)}
                className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
              />
            </div>
          </div>

          {/* Mini Map Preview */}
          {latitude && longitude && (
            <div className="rounded-lg overflow-hidden border border-border bg-muted h-32 flex items-center justify-center">
              <div className="text-center">
                <MapPin className="w-6 h-6 text-primary mx-auto mb-1" />
                <p className="text-xs text-muted-foreground">
                  {parseFloat(latitude).toFixed(4)}, {parseFloat(longitude).toFixed(4)}
                </p>
              </div>
            </div>
          )}
          
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Description (Optional)</label>
            <Input 
              placeholder="Brief description of the location" 
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
            />
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Walkable Area (m²)</label>
            <Input
              type="number"
              step="0.01"
              min="0"
              placeholder="e.g., 42.5"
              value={walkableAreaM2}
              onChange={(e) => handleWalkableAreaChange(e.target.value)}
              className="bg-secondary border-border text-foreground placeholder:text-muted-foreground"
            />
            <p className="text-xs text-muted-foreground">Used for the congestion part of the Pedestrian Traffic Severity Index.</p>
            {walkableAreaError && <p className="text-xs text-destructive">{walkableAreaError}</p>}
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium text-foreground">Pedestrian ROI JSON (Optional)</label>
              <HoverCard openDelay={100} closeDelay={120}>
                <HoverCardTrigger asChild>
                  <button
                    type="button"
                    aria-label="Show ROI JSON prompt help"
                    className="rounded-full text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <Info className="h-4 w-4" />
                  </button>
                </HoverCardTrigger>
                <HoverCardContent align="start" side="top" sideOffset={8} className="w-[min(32rem,calc(100vw-2rem))] rounded-xl p-3 text-left">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-[11px] font-medium text-foreground">Copy this prompt for AI ROI conversion:</p>
                      <Button type="button" variant="outline" size="sm" className="border-border" onClick={() => void handleCopyRoiPrompt()}>
                        {roiPromptCopied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                        {roiPromptCopied ? "Copied" : "Copy prompt"}
                      </Button>
                    </div>
                    <pre className="smooth-scrollbar max-h-72 overflow-y-auto whitespace-pre-wrap rounded-lg bg-background/15 p-3 font-mono text-[10px] leading-relaxed text-background/95 select-all">
                      {ROI_PROMPT_TEMPLATE}
                    </pre>
                  </div>
                </HoverCardContent>
              </HoverCard>
            </div>
            <Textarea
              placeholder='{"referenceSize":[1920,1080],"includePolygonsNorm":[[[0.24,0.98],[0.99,0.99],[0.22,0.09]]]}'
              value={roiCoordinatesText}
              onChange={(e) => setRoiCoordinatesText(e.target.value)}
              className="min-h-32 bg-secondary font-mono text-xs border-border text-foreground placeholder:text-muted-foreground"
            />
            <p className="text-xs text-muted-foreground">Only pedestrian foot-points inside these normalized polygons will count toward PTSI.</p>
            {roiError && <p className="text-xs text-destructive">{roiError}</p>}
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium text-foreground">Entry/Exit Points JSON (optional)</label>
              <HoverCard openDelay={100} closeDelay={120}>
                <HoverCardTrigger asChild>
                  <button
                    type="button"
                    aria-label="Show Entry/Exit Points JSON help"
                    className="rounded-full text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <Info className="h-4 w-4" />
                  </button>
                </HoverCardTrigger>
                <HoverCardContent align="start" side="top" sideOffset={8} className="w-[min(34rem,calc(100vw-2rem))] rounded-xl p-3 text-left">
                  <div className="space-y-3">
                    <div className="space-y-1.5 text-[11px] leading-relaxed text-muted-foreground">
                      <p>This field is for directional pedestrian counting in gate views. Annotate three thin strips across the pedestrian path in sequential order: strip_0, strip_1, strip_2. Then specify whether the path 0→1→2 means entering or exiting. The reverse path will be treated as the opposite direction.</p>
                      <ul className="list-disc space-y-1 pl-4">
                        <li>Provide the raw image.</li>
                        <li>Provide the image with strip annotations.</li>
                        <li>Provide the annotation file.</li>
                        <li>Provide a short description of which direction should count as entering and which should count as exiting.</li>
                      </ul>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-[11px] font-medium text-foreground">Copy this exact ChatGPT prompt:</p>
                      <Button type="button" variant="outline" size="sm" className="border-border" onClick={() => void handleCopyEntryExitPrompt()}>
                        {entryExitPromptCopied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                        {entryExitPromptCopied ? "Copied" : "Copy prompt"}
                      </Button>
                    </div>
                    <pre className="smooth-scrollbar max-h-80 overflow-y-auto whitespace-pre-wrap rounded-lg bg-background/15 p-3 font-mono text-[10px] leading-relaxed text-background/95 select-all">
                      {ENTRY_EXIT_PROMPT_TEMPLATE}
                    </pre>
                  </div>
                </HoverCardContent>
              </HoverCard>
            </div>
            <Textarea
              placeholder={'{"referenceSize":[1920,1080],"gateDirectionZonesNorm":{"strip_0":[[0.74,0.62],[0.22,0.81],[0.21,0.76],[0.71,0.59]],"strip_1":[[0.78,0.68],[0.23,0.91],[0.22,0.88],[0.76,0.66]],"strip_2":[[0.86,0.76],[0.25,1],[0.24,0.99],[0.84,0.74]]},"directionMapping":{"path_0_1_2":"entering","path_2_1_0":"exiting"}}'}
              value={entryExitPointsText}
              onChange={(e) => handleEntryExitPointsChange(e.target.value)}
              className="min-h-40 bg-secondary font-mono text-xs border-border text-foreground placeholder:text-muted-foreground"
            />
            <div className="space-y-2 text-xs text-muted-foreground">
              <p>This field is for directional pedestrian counting in gate views. Annotate three thin strips across the pedestrian path in sequential order: strip_0, strip_1, strip_2. Then specify whether path 0→1→2 means entering or exiting. The reverse path will be treated as the opposite direction.</p>
              <ul className="list-disc space-y-1 pl-4">
                <li><span className="font-medium text-foreground">strip_0</span>, <span className="font-medium text-foreground">strip_1</span>, and <span className="font-medium text-foreground">strip_2</span> are ordered strips across the pedestrian path.</li>
                <li>The actual meaning of entering vs exiting is decided through <span className="font-medium text-foreground">directionMapping.path_0_1_2</span> and <span className="font-medium text-foreground">directionMapping.path_2_1_0</span>.</li>
              </ul>
              <p>Direction is inferred from motion order: <span className="font-medium text-foreground">strip_0 -&gt; strip_1 -&gt; strip_2</span> or <span className="font-medium text-foreground">strip_2 -&gt; strip_1 -&gt; strip_0</span>.</p>
              <p>When using ChatGPT to clean the annotation, provide the raw image, the image with strip annotations, the annotation file, and a short description of which direction should count as entering and which should count as exiting.</p>
            </div>
            {entryExitPointsError && <p className="text-xs text-destructive">{entryExitPointsError}</p>}
          </div>
        </div>

        <DialogFooter className="shrink-0 px-6 pb-6 pt-2">
          <Button variant="outline" onClick={handleClose} disabled={isSubmitting} className="border-border text-foreground">
            Cancel
          </Button>
          <Button 
            onClick={() => void handleSubmit()}
            disabled={!name || !latitude || !longitude || isSubmitting}
            className="bg-primary text-primary-foreground hover:bg-primary/90"
          >
            {isSubmitting ? "Saving..." : isEditing ? "Save Changes" : "Add Location"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
