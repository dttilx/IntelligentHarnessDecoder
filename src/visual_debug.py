from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import ComponentCandidate, PageImage


def _font(size: int = 18) -> ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_marked_pages(
    page_images: list[PageImage],
    candidates: list[ComponentCandidate],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_page: dict[int, list[ComponentCandidate]] = defaultdict(list)
    for item in candidates:
        by_page[item.page_number].append(item)

    font = _font()
    for page_image in page_images:
        items = by_page.get(page_image.page_number, [])
        if not items:
            continue

        with Image.open(page_image.path) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image)
            for item in items:
                x1, y1, x2, y2 = item.box
                color = _category_color(item.category)
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                label = f"{item.name} {item.confidence:.2f}"
                label_y = max(0, y1 - 22)
                text_box = draw.textbbox((x1, label_y), label, font=font)
                draw.rectangle(text_box, fill=(255, 255, 255))
                draw.text((x1, label_y), label, fill=color, font=font)

            image.save(output_dir / f"page_{page_image.page_number:03d}_marked.jpg", quality=92)


def _category_color(category: str) -> tuple[int, int, int]:
    return {
        "connector": (0, 125, 255),
        "relay": (180, 80, 0),
        "fuse": (210, 30, 30),
        "ground": (40, 150, 70),
        "controller": (120, 50, 200),
        "component_name": (0, 150, 150),
        "reference": (220, 0, 180),
    }.get(category, (0, 0, 0))
