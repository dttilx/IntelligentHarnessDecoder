from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import OCRConfig
from .models import Box, ImageTile, OCRResult


def _load_paddle_ocr():
    # PaddleOCR on some Windows CPU environments can fail inside OneDNN/MKLDNN.
    # Disable that backend before Paddle is imported.
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_use_onednn", "0")
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "缺少 PaddleOCR。请先执行：pip install paddleocr paddlepaddle"
        ) from exc
    return PaddleOCR


def _box_from_points(points: list[list[float]], x_offset: int, y_offset: int) -> Box:
    xs = [float(point[0]) + x_offset for point in points]
    ys = [float(point[1]) + y_offset for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


class OCREngine:
    def __init__(self, config: OCRConfig) -> None:
        PaddleOCR = _load_paddle_ocr()
        self.config = config
        self.engine = PaddleOCR(
            lang=config.lang,
            use_angle_cls=config.use_angle_cls,
            use_gpu=config.use_gpu,
            show_log=config.show_log,
            enable_mkldnn=False,
            cpu_threads=4,
        )

    def recognize_tile(self, tile: ImageTile) -> list[OCRResult]:
        raw_result = self.engine.ocr(str(tile.path), cls=self.config.use_angle_cls)
        return self._parse_paddle_result(raw_result, tile)

    def recognize_tiles(self, tiles: list[ImageTile]) -> list[OCRResult]:
        results: list[OCRResult] = []
        for tile in tiles:
            results.extend(self.recognize_tile(tile))
        return results

    def _parse_paddle_result(self, raw_result: Any, tile: ImageTile) -> list[OCRResult]:
        parsed: list[OCRResult] = []
        if not raw_result:
            return parsed

        lines = raw_result[0] if len(raw_result) == 1 and isinstance(raw_result[0], list) else raw_result
        for item in lines:
            if not item or len(item) < 2:
                continue
            points = item[0]
            text_info = item[1]
            if not points or not text_info:
                continue

            text = str(text_info[0]).strip()
            try:
                confidence = float(text_info[1])
            except (TypeError, ValueError, IndexError):
                confidence = 0.0

            if not text or confidence < self.config.min_confidence:
                continue

            parsed.append(
                OCRResult(
                    page_number=tile.page_number,
                    text=text,
                    confidence=confidence,
                    box=_box_from_points(points, tile.x_offset, tile.y_offset),
                    source_image=str(tile.path),
                    tile_id=tile.tile_id,
                )
            )

        return parsed


def save_raw_ocr_csv(results: list[OCRResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    rows = [
        {
            "page": item.page_number,
            "text": item.text,
            "confidence": item.confidence,
            "x1": item.box[0],
            "y1": item.box[1],
            "x2": item.box[2],
            "y2": item.box[3],
            "tile_id": item.tile_id,
            "source_image": item.source_image,
        }
        for item in results
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
