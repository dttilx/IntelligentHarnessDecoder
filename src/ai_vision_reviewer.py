from __future__ import annotations

import base64
import hashlib
import json
import os
import re
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
    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    max_names: int = 30
    margin: int = 180
    timeout: int = 30
    api_key_env: str = ""
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

    api_key = os.environ.get(_api_key_env(config), "").strip()
    if config.offline or not api_key:
        _write_offline_prompt(evidence, vision_dir / "offline_vision_review_prompt.md")
        return []

    decisions: list[VisionDecision] = []
    cache_path = vision_dir / "vision_review_cache.jsonl"
    cache = _load_cache(cache_path) if config.use_cache else {}
    jsonl_path = vision_dir / "vision_review.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as writer:
        for index, item in enumerate(evidence, start=1):
            cache_key = _cache_key(item, config)
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
    if config.provider == "dashscope":
        return _review_one_chat_completion(evidence, api_key, config)
    if config.provider == "openai":
        return _review_one_openai_response(evidence, api_key, config)
    return _error_decision(evidence, f"未知 AI provider: {config.provider}")


def _review_one_openai_response(
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
        return _error_decision(evidence, f"HTTP {exc.code}: {message[:500]}")
    except OSError as exc:
        return _error_decision(evidence, str(exc))

    try:
        data = json.loads(body)
        text = _openai_response_text(data)
        parsed = _parse_json_text(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return _error_decision(evidence, f"模型未返回可解析 JSON: {exc}")

    return VisionDecision(
        raw_name=evidence.name,
        final_name=normalize_name(str(parsed.get("final_name") or evidence.name)),
        decision=_normalized_decision(str(parsed.get("decision") or "candidate")),
        confidence=_bounded_float(parsed.get("confidence"), default=0.5),
        reason=str(parsed.get("reason") or ""),
        page_number=evidence.page_number,
        crop_path=str(evidence.crop_path),
    )


def _review_one_chat_completion(
    evidence: VisionEvidence,
    api_key: str,
    config: VisionReviewConfig,
) -> VisionDecision:
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _vision_prompt(evidence),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(evidence.crop_path),
                        },
                    },
                ],
            }
        ],
    }
    req = request.Request(
        _provider_url(config),
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
        return _error_decision(evidence, f"HTTP {exc.code}: {message[:500]}")
    except OSError as exc:
        return _error_decision(evidence, str(exc))

    try:
        data = json.loads(body)
        text = _chat_completion_text(data)
        parsed = _parse_json_text(text)
    except (json.JSONDecodeError, ValueError) as exc:
        return _error_decision(evidence, f"模型未返回可解析 JSON: {exc}")

    return VisionDecision(
        raw_name=evidence.name,
        final_name=normalize_name(str(parsed.get("final_name") or evidence.name)),
        decision=_normalized_decision(str(parsed.get("decision") or "candidate")),
        confidence=_bounded_float(parsed.get("confidence"), default=0.5),
        reason=str(parsed.get("reason") or ""),
        page_number=evidence.page_number,
        crop_path=str(evidence.crop_path),
    )


def _api_key_env(config: VisionReviewConfig) -> str:
    if config.api_key_env:
        return config.api_key_env
    if config.provider == "dashscope":
        return "DASHSCOPE_API_KEY"
    return "OPENAI_API_KEY"


def _provider_url(config: VisionReviewConfig) -> str:
    if config.provider == "dashscope":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    return "https://api.openai.com/v1/chat/completions"


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


def _openai_response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                chunks.append(str(content.get("text", "")))
    return "\n".join(chunks).strip()


def _chat_completion_text(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("chat completion 没有 choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                chunks.append(str(item.get("text", "")))
        return "\n".join(chunks).strip()
    raise ValueError("chat completion content 格式未知")


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


def _cache_key(evidence: VisionEvidence, config: VisionReviewConfig) -> str:
    digest = hashlib.sha256(evidence.crop_path.read_bytes()).hexdigest()[:24]
    return f"{config.provider}|{config.model}|{evidence.normalized_name}|{digest}"


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
    _write_final_answer(decisions, final_dir, review_dir)


def _write_final_answer(
    decisions: list[VisionDecision],
    final_dir: Path,
    review_dir: Path,
) -> None:
    rule_names = _read_names(final_dir / "gold_names.txt")
    rows = []
    merged: dict[str, str] = {}
    suppressed: set[str] = set()
    ai_final_names: list[str] = []

    for name in rule_names:
        normalized = normalize_name(name)
        if not normalized:
            continue
        merged.setdefault(normalized, name)
        rows.append(
            {
                "name": name,
                "normalized_name": normalized,
                "source": "rule_gold",
                "raw_name": "",
                "confidence": "",
                "reason": "",
            }
        )

    for item in decisions:
        if item.decision != "accepted" or item.confidence < 0.65:
            continue
        final_name = normalize_name(item.final_name)
        if not final_name:
            continue
        normalized = normalize_name(final_name)
        raw_normalized = normalize_name(item.raw_name)
        if raw_normalized and raw_normalized != normalized:
            if _is_lossy_generalization(raw_normalized, normalized):
                final_name = item.raw_name
                normalized = raw_normalized
            else:
                suppressed.add(raw_normalized)
        ai_final_names.append(final_name)
        merged[normalized] = final_name
        rows.append(
            {
                "name": final_name,
                "normalized_name": normalized,
                "source": "ai_accepted",
                "raw_name": item.raw_name,
                "confidence": item.confidence,
                "reason": item.reason,
            }
        )

    # Keep three review layers instead of forcing one precision/recall trade-off.
    # The default answer uses the balanced layer; strict and broad are audit aids.
    strict_names = _clean_final_names(merged, suppressed, ai_final_names)
    balanced_names = _clean_balanced_final_names(merged, suppressed, ai_final_names, final_dir)
    broad_names = _clean_broad_final_names(merged, suppressed, ai_final_names, final_dir)

    (final_dir / "final_answer_strict.txt").write_text("\n".join(strict_names), encoding="utf-8")
    (final_dir / "final_answer_balanced.txt").write_text("\n".join(balanced_names), encoding="utf-8")
    (final_dir / "final_answer_broad.txt").write_text("\n".join(broad_names), encoding="utf-8")
    (final_dir / "final_answer_names.txt").write_text(
        "\n".join(balanced_names),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_excel(review_dir / "final_answer_sources.xlsx", index=False)


def _read_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


GENERIC_FINAL_SUFFIXES = (
    "控制器",
    "传感器",
    "继电器",
    "熔断器",
    "保险丝",
    "保险盒",
    "插接器",
    "电线束",
    "线束",
    "电磁阀",
    "电机",
    "开关",
    "仪表",
    "模块",
    "喇叭",
    "灯",
    "泵",
    "阀",
    "盒",
)


def _clean_final_names(
    merged: dict[str, str],
    suppressed: set[str],
    ai_final_names: list[str],
) -> list[str]:
    names = {key: value for key, value in merged.items() if key not in suppressed}
    names = _remove_composite_names(names)
    names = _remove_generic_subnames(names, ai_final_names)
    names = _remove_final_noise(names)
    names = _normalize_final_name_dict(names, allow_loose=False)
    return sorted(names.values())


def _clean_balanced_final_names(
    merged: dict[str, str],
    suppressed: set[str],
    ai_final_names: list[str],
    final_dir: Path,
) -> list[str]:
    # Balanced output keeps high-confidence rule/AI names, then admits filtered
    # recall candidates. This is the file intended for normal submission.
    names = {key: value for key, value in merged.items() if key not in suppressed}
    names = _remove_composite_names(names)
    names = _remove_hard_final_noise(names)
    names = _add_recall_boost_names(
        names,
        _read_names(final_dir / "recall_boost_names.txt"),
        limit=180,
        allow_loose=False,
    )
    names = _remove_hard_final_noise(names)
    names = _restore_ai_names(names, ai_final_names)
    names = _normalize_final_name_dict(names, allow_loose=False)
    names = _remove_hard_final_noise(names)
    return sorted(names.values())


def _clean_broad_final_names(
    merged: dict[str, str],
    suppressed: set[str],
    ai_final_names: list[str],
    final_dir: Path,
) -> list[str]:
    # Broad output is intentionally recall-heavy. It is useful as a review pool,
    # not as the clean final answer.
    names = {key: value for key, value in merged.items() if key not in suppressed}
    names = _remove_hard_final_noise(names)
    names = _add_recall_boost_names(
        names,
        _read_names(final_dir / "recall_boost_names.txt"),
        limit=420,
        allow_loose=True,
    )
    names = _remove_hard_final_noise(names)
    names = _restore_ai_names(names, ai_final_names)
    names = _normalize_final_name_dict(names, allow_loose=True)
    names = _remove_hard_final_noise(names)
    return sorted(names.values())


def _restore_ai_names(names: dict[str, str], ai_final_names: list[str]) -> dict[str, str]:
    for name in ai_final_names:
        normalized = normalize_name(name)
        if normalized and not _is_hard_final_noise_name(normalized):
            names[normalized] = name
    return names


def _add_recall_boost_names(
    names: dict[str, str],
    recall_names: list[str],
    limit: int,
    allow_loose: bool,
) -> dict[str, str]:
    # Recall candidates come from weak evidence. Add them only when their shape
    # still looks like a component name, and cap the count to avoid flooding.
    added = 0
    for name in recall_names:
        normalized = normalize_name(name)
        if not normalized or normalized in names:
            continue
        if not _looks_like_useful_recall_name(normalized, allow_loose=allow_loose):
            continue
        names[normalized] = name
        added += 1
        if added >= limit:
            break
    return names


RECALL_EXACT_NOISE = {
    "电机",
    "开关",
    "线束",
    "电线束",
    "插接器",
    "控制器",
    "传感器",
    "仪表",
    "模块",
    "灯",
    "阀",
    "泵",
    "端子",
    "保险丝",
    "继电器",
}

RECALL_TEXT_NOISE_MARKERS = (
    "线号",
    "指示",
    "供电",
    "输入",
    "输出",
    "选择",
    "默认",
    "配置",
    "式样",
    "标签",
    "长度",
    "防护",
    "颜色",
    "采用",
    "测量",
    "观测",
    "方向",
    "状态",
    "点亮",
    "熄灭",
    "安装",
    "按图示",
    "剩余",
    "未提到",
)


def _looks_like_useful_recall_name(name: str, allow_loose: bool) -> bool:
    if name in RECALL_EXACT_NOISE:
        return False
    if _is_hard_final_noise_name(name):
        return False
    if len(name) < 3 or len(name) > (30 if allow_loose else 24):
        return False
    if any(marker in name for marker in RECALL_TEXT_NOISE_MARKERS):
        return False
    if re.fullmatch(r"[\dA-Z.\- ]+", name):
        return False
    if name.count(" ") >= 2 and not allow_loose:
        return False
    if not _component_suffix(name) and not _has_connector_or_part_prefix(name):
        return False
    return True


def _has_connector_or_part_prefix(name: str) -> bool:
    return bool(
        re.search(r"\b(?:NOx|INOx|OBD|DPF|DOC|SCR|ABS|EBS|ESC|ECAS|ASR|PM|ECU|T-?BOX)\b", name, re.I)
        or re.search(r"\b(?:X|J|K|C|D)\d{1,4}[A-Z]?\b", name)
    )


FINAL_NAME_REPAIRS = (
    ("NOX", "NOx"),
    ("INOX", "INOx"),
    ("IOx传感器", "NOx传感器"),
    ("INOx传感器", "NOx传感器"),
    ("下NOx传感器", "NOx传感器 下游"),
    ("元光继电器", "远光继电器"),
    ("儿刮水继电器", "刮水继电器"),
    ("川刮水继电器", "刮水继电器"),
    ("刮冰电机", "刮水电机"),
    ("刮脉电机", "刮水电机"),
    ("古底盘电线束", "右底盘电线束"),
    ("古前ABS传感器", "右前ABS传感器"),
    ("同盘开关", "组合开关"),
    ("科右组合开关", "右组合开关"),
    ("身搭铁", "车身搭铁"),
    ("车车身搭铁", "车身搭铁"),
    ("室熔断器盒", "驾驶室熔断器盒"),
    ("室熔断器", "驾驶室熔断器"),
    ("室液压", "驾驶室液压"),
    ("动机电线束", "发动机电线束"),
    ("压传感器", "气压传感器"),
    ("压翻转", "液压翻转"),
    ("灯光旋", "灯光旋钮开关"),
    ("线束端", "线束"),
    ("开燕", "开关"),
    ("优剩叭", "喇叭"),
    ("发发动机", "发动机"),
    ("继电器儿4", "继电器"),
    ("继电器儿", "继电器"),
    ("贴点烟器照明灯", "点烟器照明灯"),
)

TRAILING_EVIDENCE_PATTERN = re.compile(
    r"\s+(?:"
    r"[XCJKD]\d{1,4}[A-Z]?|"
    r"EBS-X\d+|EBS|ABS|T-?BOX|DH1|DL1|"
    r"PP\d{4,}|DWJ-[A-Z]\d+|"
    r"\d+-\d+(?:-\d+)?"
    r")$",
    re.IGNORECASE,
)

LEADING_EVIDENCE_PATTERN = re.compile(
    r"^(?:"
    r"-?\d{1,8}[A-Z]?(?:-\d+[A-Z]?){0,2}-?|"
    r"\d+[A-Z](?:-\d+[A-Z]?){1,3}|"
    r"[A-Z]\d+[A-Z]\d+|"
    r"[A-Z]{1,4}\d{1,4}[A-Z]?-?|"
    r"[A-Z]{1,4}-[A-Z]?\d+|"
    r"EBS-X\d+|T-?BOX|ABS|EBS"
    r")\s+",
    re.IGNORECASE,
)

FINAL_TEXT_NOISE_MARKERS = (
    "摄像头",
    "天线",
    "传喇叭",
    "TBOX传",
    "TROX",
    "各用",
    "所示",
    "本体",
    "共40路",
    "无日行灯",
    "有日行灯",
    "再生请求 禁止再生开关",
    "继电器液压翻转电机",
    "传感器NOx传感器",
)

FINAL_COMPONENT_SUFFIXES = (
    "控制器",
    "传感器",
    "继电器",
    "熔断器",
    "保险丝",
    "保险丝盒",
    "插接器",
    "插座",
    "电线束",
    "线束",
    "电磁阀",
    "电机",
    "开关",
    "仪表",
    "模块",
    "喇叭",
    "雾灯",
    "灯",
    "泵",
    "阀",
    "搭铁",
)

FINAL_EXACT_NOISE = {
    "传感器",
    "控制器",
    "插接器",
    "电线束",
    "线束",
    "开关",
    "电机",
    "继电器",
    "熔断器",
    "保险丝",
    "模块",
    "仪表",
    "灯",
    "阀",
    "泵",
}


def _normalize_final_name_dict(names: dict[str, str], allow_loose: bool) -> dict[str, str]:
    # Normalize after merging so duplicate variants collapse into one final name.
    # Part numbers and connector ids are treated as evidence, not answer text.
    normalized: dict[str, str] = {}
    source_keys = set(names)
    for _, original in names.items():
        cleaned = _normalize_final_display_name(original, allow_loose=allow_loose)
        if not cleaned:
            continue
        key = normalize_name(cleaned)
        if not key or _is_final_display_noise(key, source_keys, allow_loose=allow_loose):
            continue
        previous = normalized.get(key)
        if previous is None or _prefer_final_display_name(cleaned, previous):
            normalized[key] = cleaned
    return normalized


def _normalize_final_display_name(name: str, allow_loose: bool) -> str:
    # Final display names should be human-submittable component names. Strip
    # leading/trailing evidence ids and repair common OCR fragments here.
    value = normalize_name(name)
    if not value:
        return ""
    for old, new in FINAL_NAME_REPAIRS:
        value = value.replace(old, new)
    value = re.sub(r"^[I]?\d+\.(?=[\u4e00-\u9fff])", "", value)
    value = re.sub(r"^[JK]\d+(?=[\u4e00-\u9fff])", "", value)
    value = re.sub(r"^PF(?=压差传感器)", "DPF", value)
    value = re.sub(r"^(?:DJ)?[A-Z0-9]+(?:-[A-Z0-9]+){1,4}(?=[\u4e00-\u9fff])", "", value)
    value = re.sub(r"^-路(?=底盘)", "", value)
    value = _strip_leading_evidence(value)
    value = _strip_trailing_evidence(value)
    value = _strip_leading_evidence(value)
    value = _collapse_directional_sensor_name(value)
    if not allow_loose:
        value = _drop_relation_tail(value)
    return normalize_name(value)


def _strip_leading_evidence(name: str) -> str:
    value = name.strip()
    while True:
        match = LEADING_EVIDENCE_PATTERN.match(value)
        if not match:
            return value
        candidate = value[match.end() :].strip()
        if not candidate or not _has_final_component_shape(candidate):
            return value
        value = candidate


def _strip_trailing_evidence(name: str) -> str:
    value = name.strip()
    while True:
        match = TRAILING_EVIDENCE_PATTERN.search(value)
        if not match:
            return value
        candidate = value[: match.start()].strip()
        if not candidate or not _has_final_component_shape(candidate):
            return value
        value = candidate


def _collapse_directional_sensor_name(name: str) -> str:
    if name == "x传感器":
        return "NOx传感器"
    value = re.sub(r"(?:NOx传感器){2,}\s*", "NOx传感器 ", name, flags=re.I)
    value = re.sub(r"NOx传感器\s*下(?:游)?下?(?:\s+[XC]\w*)?$", "NOx传感器 下游", value, flags=re.I)
    value = re.sub(r"NOx传感器\s*上(?:游)?上?(?:\s+[XC]\w*)?$", "NOx传感器 上游", value, flags=re.I)
    value = re.sub(r"NOx传感器\s*上(?:游)?(?:\s+X\d+[A-Z]?)?$", "NOx传感器 上游", value, flags=re.I)
    value = re.sub(r"NOx传感器\s*下(?:游)?(?:\s+X\w*)?$", "NOx传感器 下游", value, flags=re.I)
    value = re.sub(r"NOx传感器下游", "NOx传感器 下游", value, flags=re.I)
    value = re.sub(r"NOx传感器上游", "NOx传感器 上游", value, flags=re.I)
    return value


def _drop_relation_tail(name: str) -> str:
    if " " in name:
        parts = [part.strip() for part in name.split() if part.strip()]
        if len(parts) == 2 and all(_has_final_component_shape(part) for part in parts):
            return parts[0]
    for marker in (" 与", "与", " 对接", "对接"):
        if marker in name and not name.endswith("对接插接器"):
            head = name.split(marker, 1)[0].strip()
            if _has_final_component_shape(head):
                return head
    return name


def _has_final_component_shape(name: str) -> bool:
    base = re.sub(r"[-_]?\d{1,3}$", "", name)
    if _component_suffix(name) or _component_suffix(base):
        return True
    if any(name.endswith(suffix) or base.endswith(suffix) for suffix in FINAL_COMPONENT_SUFFIXES):
        return True
    return _has_connector_or_part_prefix(name)


def _prefer_final_display_name(candidate: str, previous: str) -> bool:
    if len(candidate) < len(previous):
        return True
    if any(token in previous for token in (" X", " C", " T-BOX", " EBS", " ABS")):
        return True
    return False


def _is_final_display_noise(name: str, all_names: set[str], allow_loose: bool) -> bool:
    if _is_hard_final_noise_name(name):
        return True
    if name in FINAL_EXACT_NOISE:
        return True
    if any(marker in name for marker in FINAL_TEXT_NOISE_MARKERS):
        return True
    if "请求" in name and "电线束" in name:
        return True
    if "禁止再生开关" in name and "电线束" in name:
        return True
    if name in {"控制器", "控制开关", "功能开关", "器开关", "插接器型号", "总成插接器", "成插接器", "束对接插接器"}:
        return True
    if not allow_loose and _is_relation_name_with_better_parts(name, all_names):
        return True
    if re.fullmatch(r"[\dA-Z.\- ]+", name):
        return True
    if len(name) < 3:
        return True
    return False


def _remove_composite_names(names: dict[str, str]) -> dict[str, str]:
    keys = set(names)
    result: dict[str, str] = {}
    for normalized, name in names.items():
        if _is_covered_by_shorter_names(normalized, keys):
            continue
        if _is_redundant_concatenated_name(normalized, keys):
            continue
        result[normalized] = name
    return result


def _is_covered_by_shorter_names(name: str, all_names: set[str]) -> bool:
    if " " not in name and "/" not in name and "／" not in name:
        return False
    parts = [
        normalize_name(part)
        for part in name.replace("／", "/").replace(" ", "/").split("/")
        if normalize_name(part)
    ]
    if len(parts) < 2:
        return False
    covered = 0
    for part in parts:
        if part in all_names:
            covered += 1
            continue
        if any(part in other and other != name for other in all_names):
            covered += 1
    return covered >= 2


COMPOSITE_ACTION_MARKERS = (
    "请求",
    "禁止",
    "允许",
    "控制",
    "调整",
    "调节",
    "转换",
    "翻转",
    "诊断",
    "再生",
)


def _is_lossy_generalization(raw_name: str, final_name: str) -> bool:
    if len(final_name) >= len(raw_name):
        return False
    if raw_name.startswith(final_name):
        tail = raw_name[len(final_name) :]
        return bool(tail) and all(ch.isdigit() or ch.isascii() for ch in tail)
    return False


def _is_redundant_concatenated_name(name: str, all_names: set[str]) -> bool:
    if len(name) < 8:
        return False
    if sum(1 for marker in COMPOSITE_ACTION_MARKERS if marker in name) < 2:
        return False
    suffix = _component_suffix(name)
    if not suffix:
        return False
    shorter_related = [
        other
        for other in all_names
        if other != name and len(other) < len(name) and other.endswith(suffix) and other in name
    ]
    return len(shorter_related) >= 1


def _component_suffix(name: str) -> str:
    for suffix in sorted(GENERIC_FINAL_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            return suffix
    return ""


def _remove_generic_subnames(
    names: dict[str, str],
    ai_final_names: list[str],
) -> dict[str, str]:
    ai_keys = {normalize_name(name) for name in ai_final_names}
    result: dict[str, str] = {}
    for normalized, name in names.items():
        if normalized in ai_keys:
            result[normalized] = name
            continue
        if _has_more_specific_name(normalized, set(names)):
            continue
        result[normalized] = name
    return result


def _has_more_specific_name(name: str, all_names: set[str]) -> bool:
    if not any(name.endswith(suffix) for suffix in GENERIC_FINAL_SUFFIXES):
        return False
    for other in all_names:
        if other == name:
            continue
        if len(other) <= len(name):
            continue
        if other.endswith(name) or name in other:
            return True
    return False


FINAL_NOISE_KEYWORDS = (
    "按图示",
    "安装框架",
    "特征编",
    "蓄电池选",
    "定位点",
    "输入",
)

FINAL_FRAGMENT_SUFFIXES = (
    "与",
    "对",
    "对接",
    "总",
    "总成插",
    "配置化",
)

FINAL_GARBLED_MARKERS = (
    "优剩",
    "TROX",
)


def _remove_final_noise(names: dict[str, str]) -> dict[str, str]:
    keys = set(names)
    result: dict[str, str] = {}
    for normalized, name in names.items():
        if _is_final_noise_name(normalized, keys):
            continue
        result[normalized] = name
    return result


def _is_final_noise_name(name: str, all_names: set[str]) -> bool:
    if _is_hard_final_noise_name(name):
        return True
    if _is_relation_name_with_better_parts(name, all_names):
        return True
    return False


def _remove_hard_final_noise(names: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for normalized, name in names.items():
        if _is_hard_final_noise_name(normalized):
            continue
        result[normalized] = name
    return result


def _is_hard_final_noise_name(name: str) -> bool:
    if _is_short_reference_id(name):
        return True
    if name in {"ABS", "EBS", "VCU", "搭铁", "仪表板", "对接插接器"}:
        return True
    if any(keyword in name for keyword in FINAL_NOISE_KEYWORDS):
        return True
    if any(marker in name.upper() for marker in FINAL_GARBLED_MARKERS):
        return True
    if any(name.endswith(suffix) for suffix in FINAL_FRAGMENT_SUFFIXES):
        return True
    if _is_repeated_phrase_noise(name):
        return True
    return False


def _is_short_reference_id(name: str) -> bool:
    if len(name) > 5:
        return False
    return bool(re.fullmatch(r"[A-Z]{1,3}\d{0,4}[A-Z]?", name))


def _is_repeated_phrase_noise(name: str) -> bool:
    for size in range(2, max(2, len(name) // 2 + 1)):
        for start in range(0, len(name) - size * 2 + 1):
            phrase = name[start : start + size]
            if phrase and phrase * 3 in name:
                return True
    return False


def _is_relation_name_with_better_parts(name: str, all_names: set[str]) -> bool:
    if "与" not in name and "接" not in name:
        return False
    if not any(term in name for term in ("电线束", "线束", "插接器")):
        return False
    shorter = [
        other
        for other in all_names
        if other != name and len(other) < len(name) and other in name and len(other) >= 4
    ]
    return len(shorter) >= 1
