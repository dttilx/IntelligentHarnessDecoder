from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

import pandas as pd
from PIL import Image

from .candidate_scorer import score_candidates
from .component_extractor import normalize_name, with_candidate_decisions
from .models import ComponentCandidate, PDFTextItem, PageImage, ScoredName


@dataclass(frozen=True)
class VisionReviewConfig:
    enabled: bool = False
    model: str = "gpt-4.1-mini"
    max_names: int = 30
    margin: int = 180
    timeout: int = 30
    api_key_env: str = "OPENAI_API_KEY"
    offline: bool = False
    use_cache: bool = True


@dataclass(frozen=True)
class VisionEvidence:
    name: str
    normalized_name: str
    tier: str
    score: float
    page_number: int
    crop_path: Path
    source_text: str


@dataclass(frozen=True)
class VisionDecision:
    raw_name: str
    final_name: str
    decision: str
    confidence: float
    reason: str
    page_number: int
    crop_path: str


def run_ai_vision_review(
    candidates: list[ComponentCandidate],
    pdf_text_items: list[PDFTextItem],
    page_images: list[PageImage],
    output_dir: Path,
    config: VisionReviewConfig,
) -> list[VisionDecision]:
    review_dir = output_dir / "review"
    final_dir = output_dir / "final"
    vision_dir = output_dir / "ai_vision"
    crops_dir = vision_dir / "crops"
    review_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    scored = score_candidates(with_candidate_decisions(candidates), pdf_text_items)
    selected = _selected_names(scored, config.max_names)
    evidence = _build_evidence(selected, candidates, page_images, crops_dir, config.margin)
    _write_manifest(evidence, vision_dir / "vision_manifest.csv")

    api_key = os.environ.get(config.api_key_env, "").strip()
    if config.offline or not api_key:
        _write_offline_prompt(evidence, vision_dir / "offline_vision_review_prompt.md")
        return []

    decisions: list[VisionDecision] = []
    cache_path = vision_dir / "vision_review_cache.jsonl"
    cache = _load_cache(cache_path) if config.use_cache else {}
    jsonl_path = vision_dir / "vision_review.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as writer:
        for index, item in enumerate(evidence, start=1):
            cache_key = _cache_key(item)
            if cache_key in cache:
                decision = cache[cache_key]
                print(f"  AI vision {index}/{len(evidence)} cached: {item.name}")
            else:
                print(f"  AI vision {index}/{len(evidence)}: {item.name}")
                decision = _review_one(item, api_key, config)
                if config.use_cache and not _is_fatal_error(decision):
                    _append_cache(cache_path, cache_key, decision)
            decisions.append(decision)
            writer.write(json.dumps(decision.__dict__, ensure_ascii=False) + "\n")
            if _is_fatal_error(decision):
                print(f"  AI vision stopped: {decision.reason}")
                break

    _write_decisions(decisions, review_dir, final_dir)
    return decisions


def _selected_names(scored: list[ScoredName], max_names: int) -> list[ScoredName]:
    names = [item for item in scored if item.tier in {"gold", "recall_boost"}]
    names.sort(key=lambda item: (item.tier != "gold", -item.score, item.normalized_name))
    return names[:max_names]


def _build_evidence(
    scored_names: list[ScoredName],
    candidates: list[ComponentCandidate],
    page_images: list[PageImage],
    crops_dir: Path,
    margin: int,
) -> list[VisionEvidence]:
    page_image_by_number = {item.page_number: item for item in page_images}
    candidates_by_name: dict[str, list[ComponentCandidate]] = {}
    for candidate in with_candidate_decisions(candidates):
        candidates_by_name.setdefault(candidate.normalized_name, []).append(candidate)

    evidence: list[VisionEvidence] = []
    for scored in scored_names:
        candidate = _best_candidate(candidates_by_name.get(scored.normalized_name, []))
        if candidate is None:
            continue
        page_image = page_image_by_number.get(candidate.page_number)
        if page_image is None:
            continue
        crop_path = _crop_candidate(page_image, candidate, crops_dir, margin)
        evidence.append(
            VisionEvidence(
                name=scored.name,
                normalized_name=scored.normalized_name,
                tier=scored.tier,
                score=scored.score,
                page_number=candidate.page_number,
                crop_path=crop_path,
                source_text=candidate.source_text,
            )
        )
    return evidence


def _best_candidate(candidates: list[ComponentCandidate]) -> ComponentCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.decision == "accepted", item.confidence))


def _crop_candidate(
    page_image: PageImage,
    candidate: ComponentCandidate,
    crops_dir: Path,
    margin: int,
) -> Path:
    x1, y1, x2, y2 = candidate.box
    with Image.open(page_image.path) as image:
        left = max(0, int(x1) - margin)
        top = max(0, int(y1) - margin)
        right = min(image.width, int(x2) + margin)
        bottom = min(image.height, int(y2) + margin)
        crop = image.crop((left, top, right, bottom)).convert("RGB")
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in candidate.normalized_name)[:40]
        crop_path = crops_dir / f"p{candidate.page_number:03d}_{safe_name}.jpg"
        crop.save(crop_path, format="JPEG", quality=88, optimize=True)
    return crop_path


def _review_one(
    evidence: VisionEvidence,
    api_key: str,
    config: VisionReviewConfig,
) -> VisionDecision:
    payload = {
        "model": config.model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _vision_prompt(evidence),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(evidence.crop_path),
                        "detail": "high",
                    },
                ],
            }
        ],
    }
    req = request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=config.timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="ignore")
        return _error_decision(evidence, f"HTTP {exc.code}: {message[:300]}")
    except OSError as exc:
        return _error_decision(evidence, str(exc))

    data = json.loads(body)
    text = _response_text(data)
    try:
        parsed = _parse_json_text(text)
    except ValueError:
        return _error_decision(evidence, f"模型未返回可解析 JSON: {text[:300]}")

    return VisionDecision(
        raw_name=evidence.name,
        final_name=normalize_name(str(parsed.get("final_name") or evidence.name)),
        decision=_normalized_decision(str(parsed.get("decision") or "candidate")),
        confidence=_bounded_float(parsed.get("confidence"), default=0.5),
        reason=str(parsed.get("reason") or ""),
        page_number=evidence.page_number,
        crop_path=str(evidence.crop_path),
    )


def _vision_prompt(evidence: VisionEvidence) -> str:
    return (
        "你是汽车线束图/电路图元器件名称审核助手。"
        "请只根据图片中的文字和上下文判断候选名称是否是元器件、线束、插接器、继电器、"
        "传感器、开关、灯、电机、电磁阀、搭铁点等真实名称。\n"
        "删除说明句、半截词、重复 OCR 字、纯编号、线号、配置项。"
        "如果候选有明显 OCR 错字，请修正为图中最合理的标准名称。\n"
        "只返回 JSON，不要 Markdown。\n"
        "JSON 格式："
        '{"final_name":"修正后的名称","decision":"accepted|candidate|rejected",'
        '"confidence":0.0到1.0,"reason":"简短原因"}\n'
        f"候选名称：{evidence.name}\n"
        f"规则层级：{evidence.tier}\n"
        f"规则分数：{evidence.score:.2f}\n"
        f"OCR 原文：{evidence.source_text}\n"
    )


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text", "")))
    return "\n".join(chunks).strip()


def _parse_json_text(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = value.strip("`")
        if value.lower().startswith("json"):
            value = value[4:].strip()
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no json object")
    return json.loads(value[start : end + 1])


def _normalized_decision(value: str) -> str:
    value = value.strip().lower()
    return value if value in {"accepted", "candidate", "rejected"} else "candidate"


def _bounded_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _error_decision(evidence: VisionEvidence, reason: str) -> VisionDecision:
    return VisionDecision(
        raw_name=evidence.name,
        final_name=evidence.name,
        decision="candidate",
        confidence=0.0,
        reason=f"vision_error: {reason}",
        page_number=evidence.page_number,
        crop_path=str(evidence.crop_path),
    )


def _is_fatal_error(decision: VisionDecision) -> bool:
    reason = decision.reason.lower()
    fatal_markers = (
        "insufficient_quota",
        "invalid_api_key",
        "incorrect api key",
        "unauthorized",
        "billing",
    )
    return any(marker in reason for marker in fatal_markers)


def _cache_key(evidence: VisionEvidence) -> str:
    digest = hashlib.sha256(evidence.crop_path.read_bytes()).hexdigest()[:24]
    return f"{evidence.normalized_name}|{digest}"


def _load_cache(path: Path) -> dict[str, VisionDecision]:
    cache: dict[str, VisionDecision] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = str(row["cache_key"])
            data = row["decision"]
            decision = VisionDecision(**data)
            if _is_fatal_error(decision):
                continue
            cache[key] = decision
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    return cache


def _append_cache(path: Path, cache_key: str, decision: VisionDecision) -> None:
    row = {"cache_key": cache_key, "decision": decision.__dict__}
    with path.open("a", encoding="utf-8") as writer:
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_manifest(evidence: list[VisionEvidence], path: Path) -> None:
    rows = [
        {
            "name": item.name,
            "normalized_name": item.normalized_name,
            "tier": item.tier,
            "score": item.score,
            "page": item.page_number,
            "crop_path": str(item.crop_path),
            "source_text": item.source_text,
        }
        for item in evidence
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _write_offline_prompt(evidence: list[VisionEvidence], path: Path) -> None:
    lines = [
        "# 离线 AI 视觉审核包",
        "",
        "未检测到 OPENAI_API_KEY，因此只生成裁剪图和清单。",
        "可以逐张查看 `output/ai_vision/crops/`，或设置 API key 后加 `--ai-vision-review` 重新运行。",
        "",
    ]
    for item in evidence:
        lines.append(
            f"- {item.name} | tier={item.tier} | score={item.score:.2f} | "
            f"page={item.page_number} | crop={item.crop_path}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_decisions(
    decisions: list[VisionDecision],
    review_dir: Path,
    final_dir: Path,
) -> None:
    rows = [item.__dict__ for item in decisions]
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(review_dir / "ai_vision_review.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="vision_review")
        accepted = df[df["decision"] == "accepted"] if not df.empty else df
        accepted[["final_name"]].drop_duplicates().to_excel(
            writer,
            index=False,
            sheet_name="accepted_names",
        )

    accepted_names = []
    seen: set[str] = set()
    for item in decisions:
        if item.decision != "accepted" or item.confidence < 0.65:
            continue
        normalized = normalize_name(item.final_name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        accepted_names.append(item.final_name)
    (final_dir / "ai_verified_names.txt").write_text("\n".join(accepted_names), encoding="utf-8")
