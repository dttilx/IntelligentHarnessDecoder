from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class PageImage:
    page_number: int
    path: Path
    width: int
    height: int
    rotation: int


@dataclass(frozen=True)
class ImageTile:
    page_number: int
    tile_id: str
    path: Path
    x_offset: int
    y_offset: int
    width: int
    height: int


@dataclass(frozen=True)
class OCRResult:
    page_number: int
    text: str
    confidence: float
    box: Box
    source_image: str
    tile_id: str = ""


@dataclass(frozen=True)
class ComponentCandidate:
    name: str
    normalized_name: str
    category: str
    decision: str
    page_number: int
    confidence: float
    box: Box
    source_text: str
    source_image: str
    tile_id: str = ""


@dataclass(frozen=True)
class PDFTextItem:
    page_number: int
    text: str
    box: Box
    source: str = "pdf_text"


@dataclass(frozen=True)
class ScoredName:
    name: str
    normalized_name: str
    decision: str
    tier: str
    score: float
    category: str
    evidence_count: int
    sources: str
    pages: str
    reason: str
