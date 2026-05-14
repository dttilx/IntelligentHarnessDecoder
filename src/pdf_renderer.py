from __future__ import annotations

from pathlib import Path

from PIL import Image

from .config import RenderConfig
from .models import PageImage


def _load_fitz():
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("缺少 PyMuPDF。请先执行：pip install pymupdf") from exc
    return fitz


def parse_page_spec(page_spec: str | None, total_pages: int) -> list[int]:
    if not page_spec:
        return list(range(1, total_pages + 1))

    pages: set[int] = set()
    for part in page_spec.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            pages.add(int(item))

    valid_pages = sorted(page for page in pages if 1 <= page <= total_pages)
    if not valid_pages:
        raise ValueError(f"页码范围无效：{page_spec}，PDF 共 {total_pages} 页")
    return valid_pages


class PDFRenderer:
    def __init__(self, config: RenderConfig) -> None:
        self.config = config

    def get_page_count(self, pdf_path: Path) -> int:
        fitz = _load_fitz()
        with fitz.open(pdf_path) as doc:
            return doc.page_count

    def render(
        self,
        pdf_path: Path,
        output_dir: Path,
        pages: list[int] | None = None,
    ) -> list[PageImage]:
        fitz = _load_fitz()
        output_dir.mkdir(parents=True, exist_ok=True)
        page_images: list[PageImage] = []

        with fitz.open(pdf_path) as doc:
            selected_pages = pages or list(range(1, doc.page_count + 1))
            zoom = self.config.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)

            for page_number in selected_pages:
                page = doc[page_number - 1]
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image_path = output_dir / f"page_{page_number:03d}.{self.config.image_format}"
                pix.save(image_path)

                width = pix.width
                height = pix.height
                if self.config.rotate_landscape_to_readable and width > height:
                    with Image.open(image_path) as image:
                        rotated = image.rotate(90, expand=True)
                        rotated.save(image_path)
                        width, height = rotated.size

                page_images.append(
                    PageImage(
                        page_number=page_number,
                        path=image_path,
                        width=width,
                        height=height,
                        rotation=int(page.rotation or 0),
                    )
                )

        return page_images
