from __future__ import annotations

import re

from .config import ExtractorConfig
from .models import ComponentCandidate, OCRResult


PUNCTUATION_PATTERN = re.compile(r"""[，。；;:：、/\\\[\]【】()（）{}<>《》"'`]+""")
SPACE_PATTERN = re.compile(r"\s+")
MAX_MERGED_TEXT_LENGTH = 32
MERGE_X_GAP = 180
MERGE_Y_GAP = 140
COMPONENT_SUFFIXES = (
    "控制器",
    "传感器",
    "继电器",
    "熔断器",
    "保险丝",
    "保险盒",
    "保险",
    "插接器",
    "接插件",
    "接头",
    "端子",
    "搭铁",
    "接地",
    "电源插座",
    "插座",
    "电线束",
    "线束",
    "电磁阀",
    "电机",
    "开关",
    "按钮",
    "仪表",
    "模块",
    "喇叭",
    "灯",
    "泵",
    "阀",
)
COMPONENT_NAME_PATTERN = re.compile(
    r"[A-Za-z0-9\-]{0,8}[\u4e00-\u9fffA-Za-z0-9\-]{0,16}(?:"
    + "|".join(re.escape(suffix) for suffix in COMPONENT_SUFFIXES)
    + r")(?:[-_]?\d{1,2})?"
)
BAD_CONTEXT_KEYWORDS = (
    "按QC",
    "按GB",
    "技术条件",
    "根部测量",
    "测量点",
    "观测方向",
    "分布位置",
    "保持力",
    "工作温度",
    "波纹管",
    "胶带",
    "热缩管",
    "压接",
    "标签内容",
    "所示",
    "长度尺寸",
    "选装关系",
    "没有用到",
    "盲堵",
    "出线端视",
    "焊接",
)
INCOMPLETE_NAMES = {
    "EBS-",
    "合仪表2",
    "左组合",
    "右组合",
    "喇叭转",
    "灯光旋",
    "组合仪",
    "挡开关",
    "速箱低速挡开关",
}
GENERIC_NAMES = {
    "仪表",
    "传感器",
    "线束",
    "插接器",
    "熔断器",
    "喇叭",
    "雾灯",
}
CONNECTOR_ID_PATTERN = re.compile(r"^(?:J|X|CN|XS|XP)[-_]?\d{1,4}[A-Z]?$", re.IGNORECASE)
WIRE_ID_PATTERN = re.compile(r"^(?:A|B|S|M)[-_]?\d{1,4}[A-Z]?$", re.IGNORECASE)
SHORT_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,10}$")
PART_NUMBER_FRAGMENT_PATTERN = re.compile(r"^[0-9]+(?:-[0-9]*)?$")
OCR_REPLACEMENTS = (
    ("QBD诊断插座", "OBD诊断插座"),
    ("刺叭", "喇叭"),
    ("剌叭", "喇叭"),
    ("问歇", "间歇"),
    ("由线束", "电线束"),
    ("阵身控制器", "车身控制器"),
    ("身控制器", "车身控制器"),
    ("多态开关", "多状态开关"),
    ("接左车身电线束", "左车身电线束"),
    ("接左车门电线束", "左车门电线束"),
    ("接右车身电线束", "右车身电线束"),
    ("接右车门电线束", "右车门电线束"),
)


def normalize_text(text: str) -> str:
    text = text.strip()
    text = text.replace("—", "-").replace("–", "-").replace("－", "-")
    text = text.replace("Ｏ", "O").replace("０", "0")
    for old, new in OCR_REPLACEMENTS:
        text = text.replace(old, new)
    text = SPACE_PATTERN.sub(" ", text)
    return text


def normalize_name(name: str) -> str:
    name = normalize_text(name)
    name = PUNCTUATION_PATTERN.sub("", name)
    name = _trim_leading_ocr_noise(name)
    return name.upper() if re.fullmatch(r"[A-Za-z0-9_\-]+", name) else name


def _trim_leading_ocr_noise(name: str) -> str:
    for anchor in (
        "车身控制器",
        "组合仪表",
        "OBD诊断插座",
        "EBS控制器",
        "ABS控制器",
    ):
        index = name.find(anchor)
        if index > 0:
            return name[index:]

    return re.sub(r"^\d{1,4}[A-Za-z]?(?=[\u4e00-\u9fff])", "", name).strip()


def classify_candidate(name: str, source_text: str, config: ExtractorConfig) -> str:
    upper = name.upper()
    if re.fullmatch(r"(?:X|J|CN|XS|XP)[-_]?\d{1,4}[A-Z]?", upper):
        return "connector"
    if re.fullmatch(r"K[-_]?\d{1,4}[A-Z]?", upper):
        return "relay"
    if re.fullmatch(r"F[-_]?\d{1,4}[A-Z]?", upper):
        return "fuse"
    if re.fullmatch(r"(?:GND|G[-_]?\d{1,4})", upper):
        return "ground"
    if re.fullmatch(r"(?:ECU|VCU|ABS|EBS)", upper):
        return "controller"
    if any(keyword.lower() in source_text.lower() for keyword in config.keyword_patterns):
        return "component_name"
    return "reference"


def _keyword_candidates(text: str, config: ExtractorConfig) -> list[str]:
    candidates: list[str] = []
    if _is_bad_context(text):
        return candidates

    compact = PUNCTUATION_PATTERN.sub(" ", text)
    for match in COMPONENT_NAME_PATTERN.finditer(compact):
        candidate = _clean_component_phrase(match.group(0))
        if candidate:
            candidates.append(candidate)

    for keyword in config.keyword_patterns:
        match_index = compact.lower().find(keyword.lower())
        if match_index < 0:
            continue
        start = max(0, match_index - config.keep_context_window)
        end = min(len(compact), match_index + len(keyword) + config.keep_context_window)
        snippet = compact[start:end].strip()
        snippet = SPACE_PATTERN.sub(" ", snippet)
        snippet = _clean_component_phrase(snippet)
        if snippet and len(snippet) >= config.min_text_length:
            candidates.append(snippet)

    return candidates


def _clean_component_phrase(text: str) -> str:
    value = normalize_text(text)
    value = PUNCTUATION_PATTERN.sub(" ", value)
    value = SPACE_PATTERN.sub(" ", value).strip()
    value = re.sub(r"^\d{1,4}[A-Za-z]?(?=[\u4e00-\u9fff])", "", value).strip()
    value = _trim_part_number_tail(value)

    for anchor in ("车身控制器", "组合仪表", "OBD诊断插座"):
        index = value.find(anchor)
        if index >= 0:
            return value[index:]

    if len(value) > 22:
        matches = list(COMPONENT_NAME_PATTERN.finditer(value))
        if matches:
            value = min((match.group(0) for match in matches), key=len)

    return value


def _trim_part_number_tail(value: str) -> str:
    for suffix in COMPONENT_SUFFIXES:
        marker = f"{suffix}DJ"
        marker_index = value.find(marker)
        if marker_index >= 0:
            return value[: marker_index + len(suffix)]

    return value


def _is_bad_context(text: str) -> bool:
    return any(keyword in text for keyword in BAD_CONTEXT_KEYWORDS)


def _regex_candidates(text: str, config: ExtractorConfig) -> list[str]:
    candidates: list[str] = []
    for regex in config.reference_regexes:
        candidates.extend(match.group(0) for match in regex.finditer(text))
    return candidates


def _box_union(left: ComponentCandidate | OCRResult, right: OCRResult) -> tuple[float, float, float, float]:
    return (
        min(left.box[0], right.box[0]),
        min(left.box[1], right.box[1]),
        max(left.box[2], right.box[2]),
        max(left.box[3], right.box[3]),
    )


def _box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _nearby(left: OCRResult, right: OCRResult) -> bool:
    if left.page_number != right.page_number:
        return False
    left_x, left_y = _box_center(left.box)
    right_x, right_y = _box_center(right.box)
    return abs(left_x - right_x) <= MERGE_X_GAP and abs(left_y - right_y) <= MERGE_Y_GAP


def _should_merge_text(text: str) -> bool:
    value = normalize_text(text)
    if _is_bad_context(value):
        return False
    if len(value) > MAX_MERGED_TEXT_LENGTH:
        return False
    if SHORT_LABEL_PATTERN.fullmatch(value):
        return True
    return any(keyword in value for keyword in COMPONENT_SUFFIXES)


def _is_short_label_text(text: str) -> bool:
    value = normalize_text(text)
    if PART_NUMBER_FRAGMENT_PATTERN.fullmatch(value):
        return False
    return bool(SHORT_LABEL_PATTERN.fullmatch(value))


def _has_component_suffix(text: str) -> bool:
    value = normalize_text(text)
    return any(keyword in value for keyword in COMPONENT_SUFFIXES)


def _should_merge_pair(left: OCRResult, right: OCRResult) -> bool:
    left_is_label = _is_short_label_text(left.text)
    right_is_label = _is_short_label_text(right.text)
    left_has_component = _has_component_suffix(left.text)
    right_has_component = _has_component_suffix(right.text)

    if left_has_component and right_has_component:
        return False
    return (left_is_label and right_has_component) or (right_is_label and left_has_component)


def _merged_ocr_results(ocr_results: list[OCRResult]) -> list[OCRResult]:
    by_page: dict[int, list[OCRResult]] = {}
    for result in ocr_results:
        if _should_merge_text(result.text):
            by_page.setdefault(result.page_number, []).append(result)

    merged: list[OCRResult] = []
    seen: set[tuple[int, str, int, int]] = set()
    for page_results in by_page.values():
        ordered = sorted(page_results, key=lambda item: (item.box[1], item.box[0]))
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 : index + 8]:
                if not _nearby(left, right):
                    continue
                if not _should_merge_pair(left, right):
                    continue
                combined = normalize_text(f"{left.text} {right.text}")
                if len(combined) > MAX_MERGED_TEXT_LENGTH or _is_bad_context(combined):
                    continue
                box = _box_union(left, right)
                key = (left.page_number, combined, round(box[0] / 80), round(box[1] / 80))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(
                    OCRResult(
                        page_number=left.page_number,
                        text=combined,
                        confidence=min(left.confidence, right.confidence) * 0.92,
                        box=box,
                        source_image=left.source_image,
                        tile_id=f"{left.tile_id}+{right.tile_id}",
                    )
                )

    return merged


def _is_noise(name: str) -> bool:
    value = normalize_name(name)
    if _is_rejected_name(value):
        return True
    if WIRE_ID_PATTERN.fullmatch(value):
        return True
    return False


def _is_rejected_name(name: str) -> bool:
    value = normalize_name(name)
    if len(value) < 2:
        return True
    if value in INCOMPLETE_NAMES:
        return True
    if len(value) > 24:
        return True
    if re.fullmatch(r"\d+", value):
        return True
    if re.fullmatch(r"\d{4,8}", value):
        return True
    if re.fullmatch(r"\d{1,4}[A-Z]{1,3}", value):
        return True
    if value in {"MM", "CM", "KG", "PAGE", "PDF"}:
        return True
    if _is_bad_context(value):
        return True
    return False


def _is_generic_name(name: str, all_names: set[str]) -> bool:
    if name not in GENERIC_NAMES:
        return False
    return any(name in other and name != other for other in all_names)


def decide_candidate(item: ComponentCandidate, all_names: set[str] | None = None) -> str:
    if _is_rejected_name(item.name):
        return "rejected"
    if item.decision == "candidate":
        return "candidate"
    if WIRE_ID_PATTERN.fullmatch(item.name):
        return "candidate"
    if item.category == "reference" and not CONNECTOR_ID_PATTERN.fullmatch(item.name):
        return "candidate"
    if all_names is not None and _is_generic_name(item.name, all_names):
        return "candidate"
    return "accepted"


def extract_components(
    ocr_results: list[OCRResult],
    config: ExtractorConfig,
) -> list[ComponentCandidate]:
    candidates: list[ComponentCandidate] = []

    merged_results = _merged_ocr_results(ocr_results)
    merged_result_ids = {id(result) for result in merged_results}
    all_results = ocr_results + merged_results
    for result in all_results:
        if result.confidence < config.min_component_confidence:
            continue
        text = normalize_text(result.text)
        raw_names = _regex_candidates(text, config) + _keyword_candidates(text, config)
        for raw_name in raw_names:
            name = normalize_name(raw_name)
            if _is_rejected_name(name):
                continue
            candidates.append(
                ComponentCandidate(
                    name=name,
                    normalized_name=normalize_name(name),
                    category=classify_candidate(name, text, config),
                    decision="candidate" if id(result) in merged_result_ids else "accepted",
                    page_number=result.page_number,
                    confidence=result.confidence,
                    box=result.box,
                    source_text=text,
                    source_image=result.source_image,
                    tile_id=result.tile_id,
                )
            )

    return deduplicate_candidates(candidates)


def deduplicate_candidates(
    candidates: list[ComponentCandidate],
) -> list[ComponentCandidate]:
    best: dict[tuple[int, str, int, int], ComponentCandidate] = {}
    for item in candidates:
        x1, y1, x2, y2 = item.box
        key = (item.page_number, item.normalized_name, round(x1 / 80), round(y1 / 80))
        current = best.get(key)
        if current is None or item.confidence > current.confidence:
            best[key] = item

    merged = list(best.values())
    merged.sort(key=lambda item: (item.page_number, item.category, item.normalized_name, -item.confidence))
    return merged


def unique_component_names(candidates: list[ComponentCandidate]) -> list[str]:
    return [item.name for item in unique_components_by_decision(candidates, "accepted")]


def unique_candidate_names(candidates: list[ComponentCandidate]) -> list[str]:
    return [item.name for item in unique_components_by_decision(candidates, "candidate")]


def unique_components_by_decision(
    candidates: list[ComponentCandidate],
    decision: str,
) -> list[ComponentCandidate]:
    names: dict[str, ComponentCandidate] = {}
    for item in candidates:
        existing = names.get(item.normalized_name)
        if existing is None or item.confidence > existing.confidence:
            names[item.normalized_name] = item

    all_names = {item.name for item in names.values()}
    selected = []
    for item in names.values():
        actual_decision = decide_candidate(item, all_names)
        if actual_decision != decision:
            continue
        selected.append(_replace_decision(item, actual_decision))

    return sorted(selected, key=lambda x: (x.category, x.normalized_name))


def with_candidate_decisions(candidates: list[ComponentCandidate]) -> list[ComponentCandidate]:
    all_names = {item.name for item in candidates}
    return [_replace_decision(item, decide_candidate(item, all_names)) for item in candidates]


def _replace_decision(item: ComponentCandidate, decision: str) -> ComponentCandidate:
    return ComponentCandidate(
        name=item.name,
        normalized_name=item.normalized_name,
        category=item.category,
        decision=decision,
        page_number=item.page_number,
        confidence=item.confidence,
        box=item.box,
        source_text=item.source_text,
        source_image=item.source_image,
        tile_id=item.tile_id,
    )
