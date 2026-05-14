from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .component_extractor import extract_components
from .config import AppConfig, OCRConfig, RenderConfig, TileConfig
from .image_preprocess import create_tiles
from .ocr_engine import OCREngine, save_raw_ocr_csv
from .pdf_renderer import PDFRenderer, parse_page_spec
from .result_writer import write_outputs
from .visual_debug import draw_marked_pages


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
    pages_dir = output_dir / "pages"
    tiles_dir = output_dir / "tiles"
    raw_dir = output_dir / "ocr_raw"
    marked_dir = output_dir / "marked_pages"

    renderer = PDFRenderer(config.render)
    page_count = renderer.get_page_count(pdf_path)
    selected_pages = parse_page_spec(args.pages, page_count)
    print(f"PDF: {pdf_path}")
    print(f"总页数: {page_count}，本次处理: {selected_pages}")

    print("1/5 渲染 PDF 页面...")
    page_images = renderer.render(pdf_path, pages_dir, selected_pages)

    print("2/5 切片并预处理图片...")
    tiles = []
    for page_image in page_images:
        page_tiles = create_tiles(page_image, tiles_dir, config.tile, preprocess=True)
        tiles.extend(page_tiles)
        print(f"  page {page_image.page_number}: {len(page_tiles)} 个切片")

    print("3/5 执行 PaddleOCR...")
    ocr_engine = OCREngine(config.ocr)
    ocr_results = ocr_engine.recognize_tiles(tiles)
    save_raw_ocr_csv(ocr_results, raw_dir / "ocr_raw.csv")
    print(f"  OCR 文本条数: {len(ocr_results)}")

    print("4/5 提取元器件/部件候选...")
    candidates = extract_components(ocr_results, config.extractor)
    print(f"  候选数量: {len(candidates)}")

    print("5/5 写出结果...")
    write_outputs(candidates, output_dir)
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
