from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .ai_vision_reviewer import VisionReviewConfig, run_ai_vision_review
from .component_extractor import extract_components
from .config import AppConfig, OCRConfig, RenderConfig, TileConfig
from .image_preprocess import create_tiles
from .ocr_engine import OCREngine, save_raw_ocr_csv
from .pdf_text_extractor import extract_pdf_text_items, save_pdf_text_csv
from .pdf_renderer import PDFRenderer, parse_page_spec
from .result_writer import write_outputs, write_review_outputs
from .visual_debug import draw_marked_pages
from .models import ComponentCandidate, OCRResult, PDFTextItem, PageImage


def _load_image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as image:
        return image.size


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def _box_from_row(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (_float(row, "x1"), _float(row, "y1"), _float(row, "x2"), _float(row, "y2"))


def load_raw_ocr_csv(path: Path, selected_pages: set[int]) -> list[OCRResult]:
    if not path.exists():
        raise FileNotFoundError(f"找不到 OCR 缓存：{path}")

    results: list[OCRResult] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            page_number = _int(row, "page")
            if page_number not in selected_pages:
                continue
            results.append(
                OCRResult(
                    page_number=page_number,
                    text=str(row.get("text", "")).strip(),
                    confidence=_float(row, "confidence"),
                    box=_box_from_row(row),
                    source_image=str(row.get("source_image", "")),
                    tile_id=str(row.get("tile_id", "")),
                )
            )
    return results


def load_pdf_text_csv(path: Path, selected_pages: set[int]) -> list[PDFTextItem]:
    if not path.exists():
        return []

    items: list[PDFTextItem] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            page_number = _int(row, "page")
            if page_number not in selected_pages:
                continue
            items.append(
                PDFTextItem(
                    page_number=page_number,
                    text=str(row.get("text", "")).strip(),
                    box=_box_from_row(row),
                    source=str(row.get("source", "pdf_text")),
                )
            )
    return items


def load_components_csv(path: Path, selected_pages: set[int]) -> list[ComponentCandidate]:
    if not path.exists():
        raise FileNotFoundError(f"找不到候选缓存：{path}")

    candidates: list[ComponentCandidate] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            page_number = _int(row, "page")
            if page_number not in selected_pages:
                continue
            candidates.append(
                ComponentCandidate(
                    name=str(row.get("name", "")).strip(),
                    normalized_name=str(row.get("normalized_name", "")).strip(),
                    category=str(row.get("category", "")).strip(),
                    decision=str(row.get("decision", "candidate")).strip(),
                    page_number=page_number,
                    confidence=_float(row, "confidence"),
                    box=_box_from_row(row),
                    source_text=str(row.get("source_text", "")).strip(),
                    source_image=str(row.get("source_image", "")).strip(),
                    tile_id=str(row.get("tile_id", "")).strip(),
                )
            )
    return candidates


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def load_page_images(output_dir: Path, selected_pages: list[int]) -> list[PageImage]:
    pages_dirs = (output_dir / "images" / "pages", output_dir / "pages")
    page_images: list[PageImage] = []
    for page_number in selected_pages:
        matches = []
        for pages_dir in pages_dirs:
            matches = sorted(pages_dir.glob(f"page_{page_number:03d}.*"))
            if matches:
                break
        if not matches:
            raise FileNotFoundError(f"找不到已渲染页面图：page_{page_number:03d}.*")
        path = matches[0]
        width, height = _load_image_size(path)
        page_images.append(
            PageImage(
                page_number=page_number,
                path=path,
                width=width,
                height=height,
                rotation=0,
            )
        )
    return page_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从汽车线束图 PDF 中通过 OCR 解析元器件/部件名称。",
    )
    parser.add_argument(
        "pdf",
        type=Path,
        help="PDF 文件路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="输出目录，默认 output。",
    )
    parser.add_argument(
        "--pages",
        default=None,
        help="页码范围，例如 4 或 1,3,4-6。不填则处理全部页面。",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PDF 渲染 DPI，默认 300。")
    parser.add_argument("--tile-size", type=int, default=1800, help="OCR 切片最大尺寸。")
    parser.add_argument("--overlap", type=int, default=120, help="切片重叠像素。")
    parser.add_argument("--min-confidence", type=float, default=0.35, help="OCR 最低置信度。")
    parser.add_argument("--gpu", action="store_true", help="启用 PaddleOCR GPU。")
    parser.add_argument("--no-tiles", action="store_true", help="不切片，整页 OCR。")
    parser.add_argument("--no-marked-images", action="store_true", help="不生成标注校验图。")
    parser.add_argument("--no-pdf-text", action="store_true", help="不提取 PDF 原生文本证据。")
    parser.add_argument("--no-review-output", action="store_true", help="不生成 AI 辅助审核草稿输出。")
    parser.add_argument("--score-threshold", type=float, default=0.75, help="草稿标准答案最低分数。")
    parser.add_argument("--reuse-ocr", action="store_true", help="复用 output/raw/ocr_raw.csv，跳过切片和 OCR。")
    parser.add_argument("--vision-only", action="store_true", help="只复用已有候选结果执行 AI 视觉审核。")
    parser.add_argument("--ai-vision-review", action="store_true", help="启用 AI 视觉审核候选名称。")
    parser.add_argument("--ai-provider", choices=("openai", "dashscope"), default="openai", help="AI 视觉审核服务商。")
    parser.add_argument("--ai-vision-model", default="gpt-4.1-mini", help="AI 视觉审核模型。")
    parser.add_argument("--ai-vision-max-names", type=int, default=30, help="最多送审的名称数量。")
    parser.add_argument("--ai-vision-margin", type=int, default=180, help="候选文字裁剪外扩像素。")
    parser.add_argument("--ai-vision-timeout", type=int, default=30, help="单次 AI 视觉请求超时秒数。")
    parser.add_argument("--ai-vision-offline", action="store_true", help="只生成视觉裁剪审核包，不调用 API。")
    return parser


def run(args: argparse.Namespace) -> int:
    pdf_path = args.pdf.expanduser().resolve()
    if not pdf_path.exists():
        print(f"PDF 文件不存在：{pdf_path}", file=sys.stderr)
        return 2

    config = AppConfig(
        output_dir=args.output,
        render=RenderConfig(dpi=args.dpi),
        tile=TileConfig(
            enabled=not args.no_tiles,
            max_tile_size=args.tile_size,
            overlap=args.overlap,
        ),
        ocr=OCRConfig(min_confidence=args.min_confidence, use_gpu=args.gpu),
    )

    output_dir = config.output_dir
    images_dir = output_dir / "images"
    pages_dir = images_dir / "pages"
    tiles_dir = images_dir / "tiles"
    raw_dir = output_dir / "raw"
    marked_dir = images_dir / "marked_pages"

    renderer = PDFRenderer(config.render)
    page_count = renderer.get_page_count(pdf_path)
    selected_pages = parse_page_spec(args.pages, page_count)
    selected_page_set = set(selected_pages)
    print(f"PDF: {pdf_path}")
    print(f"总页数: {page_count}，本次处理: {selected_pages}")

    if args.vision_only:
        print("vision-only: 复用已有候选和页面图，只执行 AI 视觉审核。")
        candidates = load_components_csv(
            first_existing_path(raw_dir / "components.csv", output_dir / "components.csv"),
            selected_page_set,
        )
        pdf_text_items = load_pdf_text_csv(
            first_existing_path(raw_dir / "pdf_text.csv", output_dir / "pdf_text" / "pdf_text.csv"),
            selected_page_set,
        )
        page_images = load_page_images(output_dir, selected_pages)
        decisions = run_ai_vision_review(
            candidates,
            pdf_text_items,
            page_images,
            output_dir,
            VisionReviewConfig(
                enabled=True,
                provider=args.ai_provider,
                model=args.ai_vision_model,
                max_names=args.ai_vision_max_names,
                margin=args.ai_vision_margin,
                timeout=args.ai_vision_timeout,
                offline=args.ai_vision_offline,
            ),
        )
        if decisions:
            accepted_count = sum(1 for item in decisions if item.decision == "accepted")
            print(f"  AI 视觉审核完成: accepted={accepted_count}, total={len(decisions)}")
        else:
            print("  已生成离线裁剪审核包。")
        print(f"完成。结果目录: {output_dir.resolve()}")
        return 0

    pdf_text_items = []
    if args.reuse_ocr:
        pdf_text_items = load_pdf_text_csv(
            first_existing_path(raw_dir / "pdf_text.csv", output_dir / "pdf_text" / "pdf_text.csv"),
            selected_page_set,
        )
        print(f"reuse-ocr: 复用 PDF 文本块 {len(pdf_text_items)} 条")
    elif not args.no_pdf_text:
        print("0/6 提取 PDF 原生文本证据...")
        pdf_text_items = extract_pdf_text_items(pdf_path, selected_pages)
        save_pdf_text_csv(pdf_text_items, raw_dir / "pdf_text.csv")
        print(f"  PDF 文本块: {len(pdf_text_items)}")

    need_page_images = not args.no_marked_images or args.ai_vision_review
    page_images: list[PageImage] = []
    if args.reuse_ocr:
        print("reuse-ocr: 复用 OCR 原始结果，跳过切片和 PaddleOCR。")
        ocr_results = load_raw_ocr_csv(
            first_existing_path(raw_dir / "ocr_raw.csv", output_dir / "ocr_raw" / "ocr_raw.csv"),
            selected_page_set,
        )
        print(f"  OCR 文本条数: {len(ocr_results)}")
        if need_page_images:
            try:
                page_images = load_page_images(output_dir, selected_pages)
                print("  已复用渲染页面图。")
            except FileNotFoundError:
                print("  未找到页面图，重新渲染 PDF 页面...")
                page_images = renderer.render(pdf_path, pages_dir, selected_pages)
    else:
        print("1/6 渲染 PDF 页面...")
        page_images = renderer.render(pdf_path, pages_dir, selected_pages)

        print("2/6 切片并预处理图片...")
        tiles = []
        for page_image in page_images:
            page_tiles = create_tiles(page_image, tiles_dir, config.tile, preprocess=True)
            tiles.extend(page_tiles)
            print(f"  page {page_image.page_number}: {len(page_tiles)} 个切片")

        print("3/6 执行 PaddleOCR...")
        ocr_engine = OCREngine(config.ocr)
        ocr_results = ocr_engine.recognize_tiles(tiles)
        save_raw_ocr_csv(ocr_results, raw_dir / "ocr_raw.csv")
        print(f"  OCR 文本条数: {len(ocr_results)}")

    print("4/6 提取元器件/部件候选...")
    candidates = extract_components(ocr_results, config.extractor)
    print(f"  候选数量: {len(candidates)}")

    print("5/6 写出结果...")
    write_outputs(candidates, output_dir)
    if not args.no_review_output:
        print("6/6 生成 AI 辅助审核草稿...")
        write_review_outputs(candidates, pdf_text_items, output_dir, threshold=args.score_threshold)
    if args.ai_vision_review:
        print("AI 视觉审核候选名称...")
        decisions = run_ai_vision_review(
            candidates,
            pdf_text_items,
            page_images,
            output_dir,
            VisionReviewConfig(
                enabled=True,
                provider=args.ai_provider,
                model=args.ai_vision_model,
                max_names=args.ai_vision_max_names,
                margin=args.ai_vision_margin,
                timeout=args.ai_vision_timeout,
                offline=args.ai_vision_offline,
            ),
        )
        if decisions:
            accepted_count = sum(1 for item in decisions if item.decision == "accepted")
            print(f"  AI 视觉审核完成: accepted={accepted_count}, total={len(decisions)}")
        else:
            print("  已生成离线裁剪审核包。")
    if not args.no_marked_images:
        draw_marked_pages(page_images, candidates, marked_dir)

    print(f"完成。结果目录: {output_dir.resolve()}")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
