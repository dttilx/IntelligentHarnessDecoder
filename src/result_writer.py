from __future__ import annotations

from pathlib import Path

import pandas as pd

from .candidate_scorer import accepted_draft_names, names_by_tier, score_candidates
from .component_extractor import unique_candidate_names, unique_component_names, with_candidate_decisions
from .models import ComponentCandidate, PDFTextItem, ScoredName


def candidates_to_dataframe(candidates: list[ComponentCandidate]) -> pd.DataFrame:
    rows = [
        {
            "page": item.page_number,
            "name": item.name,
            "normalized_name": item.normalized_name,
            "category": item.category,
            "decision": item.decision,
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


def scored_names_to_dataframe(scored_names: list[ScoredName]) -> pd.DataFrame:
    rows = [
        {
            "name": item.name,
            "normalized_name": item.normalized_name,
            "decision": item.decision,
            "tier": item.tier,
            "score": item.score,
            "category": item.category,
            "evidence_count": item.evidence_count,
            "sources": item.sources,
            "pages": item.pages,
            "reason": item.reason,
        }
        for item in scored_names
    ]
    return pd.DataFrame(rows)


def write_outputs(candidates: list[ComponentCandidate], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    final_dir = output_dir / "final"
    raw_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / "components.csv"
    xlsx_path = output_dir / "components.xlsx"
    txt_path = final_dir / "components.txt"
    candidate_txt_path = final_dir / "components_candidates.txt"
    decided_candidates = with_candidate_decisions(candidates)

    df = candidates_to_dataframe(decided_candidates)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="components")
        pd.DataFrame({"name": unique_component_names(decided_candidates)}).to_excel(
            writer,
            index=False,
            sheet_name="unique_names",
        )
        pd.DataFrame({"name": unique_candidate_names(decided_candidates)}).to_excel(
            writer,
            index=False,
            sheet_name="candidate_names",
        )

    names = unique_component_names(decided_candidates)
    candidate_names = unique_candidate_names(decided_candidates)
    txt_path.write_text("\n".join(names), encoding="utf-8")
    candidate_txt_path.write_text("\n".join(candidate_names), encoding="utf-8")
    (output_dir / "components.txt").write_text("\n".join(names), encoding="utf-8")
    (output_dir / "components_candidates.txt").write_text("\n".join(candidate_names), encoding="utf-8")


def write_review_outputs(
    candidates: list[ComponentCandidate],
    pdf_text_items: list[PDFTextItem],
    output_dir: Path,
    threshold: float = 0.75,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_names = score_candidates(with_candidate_decisions(candidates), pdf_text_items)
    scored_df = scored_names_to_dataframe(scored_names)
    draft_names = accepted_draft_names(scored_names, threshold=threshold)
    recall_boost_names = names_by_tier(scored_names, "recall_boost")
    candidate_names = names_by_tier(scored_names, "candidate")
    rejected_names = names_by_tier(scored_names, "rejected")

    final_dir = output_dir / "final"
    review_dir = output_dir / "review"
    final_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    review_xlsx_path = review_dir / "components_review.xlsx"
    draft_txt_path = final_dir / "gold_names.txt"
    recall_txt_path = final_dir / "recall_boost_names.txt"
    rejected_txt_path = final_dir / "rejected_names.txt"
    draft_xlsx_path = review_dir / "draft_gold_names.xlsx"
    ai_prompt_path = review_dir / "ai_review_prompt.md"

    with pd.ExcelWriter(review_xlsx_path, engine="openpyxl") as writer:
        scored_df.to_excel(writer, index=False, sheet_name="review")
        pd.DataFrame({"name": draft_names}).to_excel(writer, index=False, sheet_name="draft_gold")
        pd.DataFrame({"name": recall_boost_names}).to_excel(writer, index=False, sheet_name="recall_boost")
        pd.DataFrame({"name": candidate_names}).to_excel(writer, index=False, sheet_name="candidate")

    draft_txt_path.write_text("\n".join(draft_names), encoding="utf-8")
    recall_txt_path.write_text("\n".join(recall_boost_names), encoding="utf-8")
    rejected_txt_path.write_text("\n".join(rejected_names), encoding="utf-8")
    with pd.ExcelWriter(draft_xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame({"name": draft_names}).to_excel(writer, index=False, sheet_name="draft_gold")
        pd.DataFrame({"name": recall_boost_names}).to_excel(writer, index=False, sheet_name="recall_boost")
    ai_prompt_path.write_text(_build_ai_review_prompt(scored_names), encoding="utf-8")
    (output_dir / "components_review.xlsx").write_bytes(review_xlsx_path.read_bytes())
    (output_dir / "draft_gold_names.txt").write_text("\n".join(draft_names), encoding="utf-8")
    (output_dir / "ai_review_prompt.md").write_text(ai_prompt_path.read_text(encoding="utf-8"), encoding="utf-8")


def _build_ai_review_prompt(scored_names: list[ScoredName]) -> str:
    lines = [
        "# AI 元器件名称审核任务",
        "",
        "请从下面候选中整理高可信元器件/线束/连接器/继电器/传感器/开关名称。",
        "",
        "规则：",
        "- 保留真实元器件、电气部件、线束、连接器、继电器、熔断器、传感器、开关、灯、电机、电磁阀、搭铁点。",
        "- 删除车型配置、说明句、信号说明、纯编号、OCR 半截词和明显乱码。",
        "- 能明显判断的 OCR 错字请归一化，例如 NOX/NOx 统一为 NOx，DPE/OPF 压差传感器统一为 DPF 压差传感器。",
        "- 输出三列：final_name, decision, reason。decision 只能是 accepted/candidate/rejected。",
        "",
    ]
    for title, tier in (
        ("high_confidence", "gold"),
        ("recall_boost", "recall_boost"),
        ("candidate", "candidate"),
    ):
        lines.extend(["", f"## {title}", ""])
        for item in scored_names:
            if item.tier != tier:
                continue
            lines.append(
                f"- name={item.name} | score={item.score:.2f} | tier={item.tier} | "
                f"sources={item.sources} | pages={item.pages} | reason={item.reason}"
            )
    return "\n".join(lines)
