from __future__ import annotations

from pathlib import Path

import pandas as pd

from .models import PDFTextItem
from .pdf_renderer import _load_fitz


def extract_pdf_text_items(pdf_path: Path, pages: list[int]) -> list[PDFTextItem]:
    fitz = _load_fitz()
    items: list[PDFTextItem] = []

    with fitz.open(pdf_path) as doc:
        for page_number in pages:
            page = doc[page_number - 1]
            for block in page.get_text("blocks"):
                if len(block) < 5:
                    continue
                x1, y1, x2, y2, text = block[:5]
                cleaned = " ".join(str(text).split())
                if not cleaned:
                    continue
                items.append(
                    PDFTextItem(
                        page_number=page_number,
                        text=cleaned,
                        box=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )

    return items


def save_pdf_text_csv(items: list[PDFTextItem], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "page": item.page_number,
            "text": item.text,
            "x1": item.box[0],
            "y1": item.box[1],
            "x2": item.box[2],
            "y2": item.box[3],
            "source": item.source,
        }
        for item in items
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
