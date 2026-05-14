from __future__ import annotations

from pathlib import Path

import pandas as pd

from .component_extractor import unique_component_names
from .models import ComponentCandidate


def candidates_to_dataframe(candidates: list[ComponentCandidate]) -> pd.DataFrame:
    rows = [
        {
            "page": item.page_number,
            "name": item.name,
            "normalized_name": item.normalized_name,
            "category": item.category,
            "confidence": item.confidence,
            "x1": item.box[0],
            "y1": item.box[1],
            "x2": item.box[2],
            "y2": item.box[3],
            "source_text": item.source_text,
            "tile_id": item.tile_id,
            "source_image": item.source_image,
        }
        for item in candidates
    ]
    return pd.DataFrame(rows)


def write_outputs(candidates: list[ComponentCandidate], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = candidates_to_dataframe(candidates)
    csv_path = output_dir / "components.csv"
    xlsx_path = output_dir / "components.xlsx"
    txt_path = output_dir / "components.txt"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="components")
        pd.DataFrame({"name": unique_component_names(candidates)}).to_excel(
            writer,
            index=False,
            sheet_name="unique_names",
        )

    names = unique_component_names(candidates)
    txt_path.write_text("\n".join(names), encoding="utf-8")
