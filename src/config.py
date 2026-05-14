from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Pattern


@dataclass(frozen=True)
class OCRConfig:
    lang: str = "ch"
    use_angle_cls: bool = True
    use_gpu: bool = False
    show_log: bool = False
    min_confidence: float = 0.35


@dataclass(frozen=True)
class RenderConfig:
    dpi: int = 300
    image_format: str = "png"
    rotate_landscape_to_readable: bool = False


@dataclass(frozen=True)
class TileConfig:
    enabled: bool = True
    max_tile_size: int = 1800
    overlap: int = 120
    min_tile_size: int = 500


@dataclass(frozen=True)
class ExtractorConfig:
    min_text_length: int = 2
    min_component_confidence: float = 0.45
    keep_context_window: int = 12
    keyword_patterns: tuple[str, ...] = (
        "控制器",
        "ECU",
        "VCU",
        "ABS",
        "EBS",
        "CAN",
        "仪表",
        "传感器",
        "开关",
        "继电器",
        "保险",
        "熔断",
        "插接器",
        "接插件",
        "接头",
        "端子",
        "针脚",
        "搭铁",
        "接地",
        "电源",
        "线束",
        "电磁阀",
        "泵",
        "电机",
        "灯",
        "喇叭",
        "模块",
        "阀",
        "按钮",
        "组合",
        "诊断",
        "尿素",
        "AdBlue",
    )
    reference_regexes: tuple[Pattern[str], ...] = field(
        default_factory=lambda: tuple(
            re.compile(pattern, re.IGNORECASE)
            for pattern in (
                r"\b(?:X|J|CN|XS|XP|K|F|S|G|M|B|U|A|P|Q|D|L|R|C|Y|T|TP|ECU|VCU)[-_]?\d{1,4}[A-Z]?\b",
                r"\b(?:GND|ACC|IGN|BAT|BATT|KL15|KL30|CANH|CANL|CAN_H|CAN_L)\b",
                r"\b\d{1,3}[A-Z]{1,3}\b",
            )
        )
    )


@dataclass(frozen=True)
class AppConfig:
    output_dir: Path = Path("output")
    render: RenderConfig = RenderConfig()
    tile: TileConfig = TileConfig()
    ocr: OCRConfig = OCRConfig()
    extractor: ExtractorConfig = ExtractorConfig()
