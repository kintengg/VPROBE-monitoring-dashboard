from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class VideoCard(BaseModel):
    id: str
    timestamp: str
    date: str
    startTime: str
    endTime: str
    pedestrianCount: int = 0
    rawPath: Optional[str] = None
    processedPath: Optional[str] = None


class LocationCreate(BaseModel):
    name: str
    latitude: float
    longitude: float
    description: str = ""
    address: str = ""
    roiCoordinates: Optional[dict[str, Any]] = None
    walkableAreaM2: Optional[float] = None


class LocationRecord(LocationCreate):
    id: str
    videos: list[VideoCard] = Field(default_factory=list)


class LocationSearchResult(BaseModel):
    name: str
    address: str = ""
    latitude: float
    longitude: float
    placeId: Optional[str] = None
    types: list[str] = Field(default_factory=list)


class VideoRecord(BaseModel):
    id: str
    locationId: str
    location: str
    timestamp: str
    date: str
    startTime: str
    endTime: str
    gpsLat: float
    gpsLng: float
    pedestrianCount: int
    rawPath: Optional[str] = None
    processedPath: Optional[str] = None


class EventRecord(BaseModel):
    id: str
    type: Literal["detection", "alert", "motion"]
    location: str
    timestamp: str
    description: str
    videoId: Optional[str] = None
    pedestrianId: Optional[int] = None
    frame: Optional[int] = None
    offsetSeconds: Optional[float] = None
    occlusionClass: Optional[int] = None


class DashboardSummary(BaseModel):
    totalUniquePedestrians: int
    averageFps: float
    totalHeavyOcclusions: int
    monitoredLocations: int


class LocationTotal(BaseModel):
    location: str
    totalPedestrians: int


class TrafficResponse(BaseModel):
    timeRange: str
    series: list[dict[str, object]] = Field(default_factory=list)
    bucketMinutes: int = 60
    zoomLevel: int = 0
    canZoomIn: bool = False
    isDrilldown: bool = False
    focusTime: Optional[str] = None
    windowStart: Optional[str] = None
    windowEnd: Optional[str] = None
    locationTotals: list[LocationTotal] = Field(default_factory=list)


class PTSITrendResponse(BaseModel):
    timeRange: str
    series: list[dict[str, object]] = Field(default_factory=list)
    bucketMinutes: int = 60
    zoomLevel: int = 0
    canZoomIn: bool = False
    isDrilldown: bool = False
    focusTime: Optional[str] = None
    windowStart: Optional[str] = None
    windowEnd: Optional[str] = None


class PTSIHourScore(BaseModel):
    hour: str
    score: float
    mode: Optional[Literal["strict-fhwa", "roi-testing"]] = None
    averagePedestrians: Optional[float] = None
    uniquePedestrians: Optional[int] = None
    occlusionMix: Optional[dict[str, float]] = None
    los: Optional[Literal["A", "B", "C", "D", "E", "F"]] = None
    losDescription: Optional[str] = None


class PTSILocation(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    hasFootage: bool
    hasPTSIData: bool
    score: Optional[float] = None
    state: Literal["clear", "moderate", "severe", "no-footage", "no-data"]
    mode: Optional[Literal["strict-fhwa", "roi-testing"]] = None
    averagePedestrians: Optional[float] = None
    uniquePedestrians: Optional[int] = None
    occlusionMix: Optional[dict[str, float]] = None
    los: Optional[Literal["A", "B", "C", "D", "E", "F"]] = None
    losDescription: Optional[str] = None
    peakHour: Optional[str] = None
    peakHourScore: Optional[float] = None
    offPeakHour: Optional[str] = None
    offPeakHourScore: Optional[float] = None
    hourlyScores: list[PTSIHourScore] = Field(default_factory=list)


class PTSIMapResponse(BaseModel):
    timeRange: str
    availableHours: list[str] = Field(default_factory=list)
    locations: list[PTSILocation] = Field(default_factory=list)


OcclusionTrendResponse = PTSITrendResponse
OcclusionHourScore = PTSIHourScore
OcclusionLocation = PTSILocation
OcclusionMapResponse = PTSIMapResponse


class AIBadge(BaseModel):
    label: str
    value: str
    tone: Literal["blue", "green", "orange", "purple", "red"]


class AISection(BaseModel):
    title: str
    body: str
    badges: list[AIBadge] = Field(default_factory=list)


class AISynthesisResponse(BaseModel):
    date: str
    timeRange: str
    sections: list[AISection]


class SearchResult(BaseModel):
    id: str
    videoId: str
    timestamp: str
    date: str
    location: str
    confidence: int
    matchReason: str
    pedestrianId: Optional[int] = None
    frame: Optional[int] = None
    offsetSeconds: Optional[float] = None
    firstTimestamp: Optional[str] = None
    lastTimestamp: Optional[str] = None
    firstOffsetSeconds: Optional[float] = None
    lastOffsetSeconds: Optional[float] = None
    thumbnailPath: Optional[str] = None
    previewPath: Optional[str] = None
    appearanceSummary: Optional[str] = None
    visualLabels: list[str] = Field(default_factory=list)
    visualObjects: list[str] = Field(default_factory=list)
    visualLogos: list[str] = Field(default_factory=list)
    visualText: list[str] = Field(default_factory=list)
    visualSummary: Optional[str] = None
    semanticScore: Optional[float] = None
    possibleMatch: bool = False
    matchStrategy: Optional[Literal["semantic", "metadata", "event"]] = None


class ModelInfo(BaseModel):
    currentModel: Optional[str] = None
    uploadedAt: Optional[str] = None


class InferenceStatus(BaseModel):
    installed: bool
    version: Optional[str] = None
    preferredTag: str
    fallbackTag: str
    currentModel: Optional[str] = None
    modelPath: Optional[str] = None
    modelExists: bool
    ready: bool


class VideoUploadStatus(BaseModel):
    uploadId: str
    state: Literal["queued", "processing", "complete", "error", "cancelled"]
    progressPercent: Optional[int] = None
    message: str
    phase: Optional[Literal["queued", "tracking", "vision", "ptsi", "finalizing"]] = None
    videoId: Optional[str] = None
    error: Optional[str] = None
    updatedAt: str
