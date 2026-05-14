from __future__ import annotations

from dataclasses import dataclass
import re

from .component_extractor import (
    CONNECTOR_ID_PATTERN,
    GENERIC_NAMES,
    PURE_CONNECTOR_ID_PATTERN,
    WIRE_ID_PATTERN,
    _is_bad_context,
    _is_config_text,
    normalize_name,
)
from .models import ComponentCandidate, PDFTextItem, ScoredName


COMPONENT_TERMS = (
    "传感器",
    "控制器",
    "继电器",
    "熔断器",
    "保险丝",
    "保险盒",
    "插接器",
    "插座",
    "电线束",
    "线束",
    "电磁阀",
    "电机",
    "开关",
    "搭铁",
    "灯",
    "模块",
    "泵",
    "阀",
)
SIGNAL_TERMS = (
    "信号",
    "电源",
    "CAN线",
    "输入",
    "输出",
    "供电",
    "点亮",
    "熄灭",
)
OCR_LIKE_NOISE = re.compile(r"[木桃闽闻婴凯圳脉]{1}")


@dataclass
class EvidenceBucket:
    name: str
    normalized_name: str
    category: str
    sources: set[str]
    pages: set[int]
    confidences: list[float]
    source_texts: list[str]
    decisions: set[str]


def score_candidates(
    candidates: list[ComponentCandidate],
    pdf_text_items: list[PDFTextItem] | None = None,
) -> list[ScoredName]:
    buckets: dict[str, EvidenceBucket] = {}

    for item in candidates:
        _add_evidence(
            buckets,
            name=item.name,
            category=item.category,
            source="ocr",
            page=item.page_number,
            confidence=item.confidence,
            source_text=item.source_text,
            decision=item.decision,
        )

    for item in pdf_text_items or []:
        for name in _names_from_pdf_text(item.text):
            _add_evidence(
                buckets,
                name=name,
                category="pdf_text",
                source="pdf_text",
                page=item.page_number,
                confidence=1.0,
                source_text=item.text,
                decision="candidate",
            )

    scored = [_score_bucket(bucket) for bucket in buckets.values()]
    scored.sort(key=lambda item: (-item.score, item.normalized_name))
    return scored


def accepted_draft_names(scored: list[ScoredName], threshold: float = 0.75) -> list[str]:
    return [
        item.name
        for item in scored
        if item.score >= threshold and item.decision == "accepted"
    ]


def _add_evidence(
    buckets: dict[str, EvidenceBucket],
    name: str,
    category: str,
    source: str,
    page: int,
    confidence: float,
    source_text: str,
    decision: str,
) -> None:
    normalized_name = normalize_name(name)
    if not normalized_name:
        return
    bucket = buckets.get(normalized_name)
    if bucket is None:
        bucket = EvidenceBucket(
            name=name,
            normalized_name=normalized_name,
            category=category,
            sources=set(),
            pages=set(),
            confidences=[],
            source_texts=[],
            decisions=set(),
        )
        buckets[normalized_name] = bucket

    bucket.sources.add(source)
    bucket.pages.add(page)
    bucket.confidences.append(confidence)
    bucket.decisions.add(decision)
    if len(bucket.source_texts) < 3:
        bucket.source_texts.append(source_text)
    if len(name) > len(bucket.name) and len(name) <= 24:
        bucket.name = name


def _names_from_pdf_text(text: str) -> list[str]:
    value = " ".join(text.split())
    if _is_bad_context(value) or _is_config_text(value):
        return []
    names = []
    for term in COMPONENT_TERMS:
        pattern = re.compile(rf"[\u4e00-\u9fffA-Za-z0-9&／/ -]{{0,14}}{re.escape(term)}(?:[-_]?\d{{1,2}})?")
        for match in pattern.finditer(value):
            name = normalize_name(match.group(0).strip())
            if 2 <= len(name) <= 24:
                names.append(name)
    return names


def _score_bucket(bucket: EvidenceBucket) -> ScoredName:
    score = 0.2
    reasons = []
    name = normalize_name(bucket.name)
    joined_text = " ".join(bucket.source_texts)

    mean_confidence = sum(bucket.confidences) / max(len(bucket.confidences), 1)
    score += min(mean_confidence, 1.0) * 0.25
    reasons.append(f"mean_conf={mean_confidence:.2f}")

    if "pdf_text" in bucket.sources:
        score += 0.18
        reasons.append("pdf_text")
    if "ocr" in bucket.sources:
        score += 0.1
        reasons.append("ocr")
    if len(bucket.pages) > 1:
        score += min(len(bucket.pages), 4) * 0.04
        reasons.append(f"pages={len(bucket.pages)}")
    if len(bucket.confidences) > 1:
        score += min(len(bucket.confidences), 5) * 0.02
        reasons.append(f"evidence={len(bucket.confidences)}")
    if any(term in name for term in COMPONENT_TERMS):
        score += 0.18
        reasons.append("component_term")
    if name in GENERIC_NAMES:
        score -= 0.2
        reasons.append("generic")
    if WIRE_ID_PATTERN.fullmatch(name) or PURE_CONNECTOR_ID_PATTERN.fullmatch(name):
        score -= 0.25
        reasons.append("pure_id")
    if CONNECTOR_ID_PATTERN.fullmatch(name):
        score -= 0.1
        reasons.append("connector_id")
    if _is_bad_context(name) or _is_bad_context(joined_text):
        score -= 0.35
        reasons.append("context_noise")
    if _is_config_text(name) or _is_config_text(joined_text):
        score -= 0.35
        reasons.append("config")
    if any(term in name for term in SIGNAL_TERMS) and not any(term in name for term in ("电源插座", "电源总开关")):
        score -= 0.12
        reasons.append("signal_like")
    if OCR_LIKE_NOISE.search(name):
        score -= 0.12
        reasons.append("ocr_noise_chars")
    if len(name) < 3:
        score -= 0.15
        reasons.append("too_short")
    if len(name) > 18:
        score -= 0.1
        reasons.append("long")

    score = max(0.0, min(score, 1.0))
    decision = "accepted" if score >= 0.75 else "candidate" if score >= 0.45 else "rejected"

    return ScoredName(
        name=bucket.name,
        normalized_name=bucket.normalized_name,
        decision=decision,
        score=round(score, 4),
        category=bucket.category,
        evidence_count=len(bucket.confidences),
        sources=",".join(sorted(bucket.sources)),
        pages=",".join(str(page) for page in sorted(bucket.pages)),
        reason="; ".join(reasons),
    )
