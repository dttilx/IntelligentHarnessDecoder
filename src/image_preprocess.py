from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from .config import TileConfig
from .models import ImageTile, PageImage


def preprocess_image(input_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image:
        image = image.convert("L")
        image = ImageOps.autocontrast(image)
        image = ImageEnhance.Contrast(image).enhance(1.35)
        image = image.filter(ImageFilter.SHARPEN)
        image.save(output_path)
    return output_path


def _positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]

    step = max(tile_size - overlap, 1)
    values = list(range(0, length, step))
    if values[-1] + tile_size < length:
        values.append(length - tile_size)
    values = [min(value, max(length - tile_size, 0)) for value in values]
    return sorted(set(values))


def create_tiles(
    page_image: PageImage,
    output_dir: Path,
    config: TileConfig,
    preprocess: bool = True,
) -> list[ImageTile]:
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(page_image.path) as image:
        width, height = image.size
        if not config.enabled or max(width, height) <= config.max_tile_size:
            tile_path = output_dir / f"page_{page_image.page_number:03d}_tile_000.png"
            if preprocess:
                preprocess_image(page_image.path, tile_path)
            else:
                image.save(tile_path)
            return [
                ImageTile(
                    page_number=page_image.page_number,
                    tile_id="000",
                    path=tile_path,
                    x_offset=0,
                    y_offset=0,
                    width=width,
                    height=height,
                )
            ]

        x_values = _positions(width, config.max_tile_size, config.overlap)
        y_values = _positions(height, config.max_tile_size, config.overlap)
        tiles: list[ImageTile] = []
        tile_index = 0

        for y in y_values:
            for x in x_values:
                right = min(x + config.max_tile_size, width)
                bottom = min(y + config.max_tile_size, height)
                if right - x < config.min_tile_size or bottom - y < config.min_tile_size:
                    continue

                crop = image.crop((x, y, right, bottom))
                tile_id = f"{tile_index:03d}"
                tile_path = output_dir / f"page_{page_image.page_number:03d}_tile_{tile_id}.png"
                if preprocess:
                    crop = crop.convert("L")
                    crop = ImageOps.autocontrast(crop)
                    crop = ImageEnhance.Contrast(crop).enhance(1.35)
                    crop = crop.filter(ImageFilter.SHARPEN)
                crop.save(tile_path)

                tiles.append(
                    ImageTile(
                        page_number=page_image.page_number,
                        tile_id=tile_id,
                        path=tile_path,
                        x_offset=x,
                        y_offset=y,
                        width=right - x,
                        height=bottom - y,
                    )
                )
                tile_index += 1

    return tiles
