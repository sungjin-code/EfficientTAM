"""Create tiny synthetic datasets for EfficientTAM training smoke tests.

This script intentionally uses only the Python standard library so `_test_train.sh`
can prepare data before ML dependencies are installed.

The generated folders match the training loaders:

Stage 1:
    {out_root}/image/images/*.png
    {out_root}/image/masks/*.png

Stage 2:
    {out_root}/video/JPEGImages/{video_id}/*.jpg
    {out_root}/video/Annotations/{video_id}/*.png

Video frames are PPM bytes saved with a `.jpg` suffix. PIL, used by the training
loader, detects the image format from file headers rather than the suffix.
"""

from __future__ import annotations

import argparse
import math
import shutil
import struct
import zlib
from pathlib import Path


Color = tuple[int, int, int]


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _write_png_rgb(path: Path, pixels: list[list[Color]]) -> None:
    height = len(pixels)
    width = len(pixels[0])
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend((r, g, b))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _write_png_gray(path: Path, pixels: list[list[int]]) -> None:
    height = len(pixels)
    width = len(pixels[0])
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        raw.extend(max(0, min(255, v)) for v in row)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _write_ppm(path: Path, pixels: list[list[Color]]) -> None:
    height = len(pixels)
    width = len(pixels[0])
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    raw = bytearray()
    for row in pixels:
        for r, g, b in row:
            raw.extend((r, g, b))
    path.write_bytes(header + bytes(raw))


def _rgb_canvas(size: int, color: Color) -> list[list[Color]]:
    return [[color for _ in range(size)] for _ in range(size)]


def _gray_canvas(size: int, value: int = 0) -> list[list[int]]:
    return [[value for _ in range(size)] for _ in range(size)]


def _draw_rect_rgb(pixels: list[list[Color]], box: tuple[int, int, int, int], color: Color) -> None:
    x0, y0, x1, y1 = box
    for y in range(max(0, y0), min(len(pixels), y1 + 1)):
        for x in range(max(0, x0), min(len(pixels[0]), x1 + 1)):
            pixels[y][x] = color


def _draw_rect_gray(pixels: list[list[int]], box: tuple[int, int, int, int], value: int) -> None:
    x0, y0, x1, y1 = box
    for y in range(max(0, y0), min(len(pixels), y1 + 1)):
        for x in range(max(0, x0), min(len(pixels[0]), x1 + 1)):
            pixels[y][x] = value


def _draw_ellipse_rgb(pixels: list[list[Color]], box: tuple[int, int, int, int], color: Color) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = max(1.0, (x1 - x0) / 2.0)
    ry = max(1.0, (y1 - y0) / 2.0)
    for y in range(max(0, y0), min(len(pixels), y1 + 1)):
        for x in range(max(0, x0), min(len(pixels[0]), x1 + 1)):
            if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
                pixels[y][x] = color


def _draw_ellipse_gray(pixels: list[list[int]], box: tuple[int, int, int, int], value: int) -> None:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    rx = max(1.0, (x1 - x0) / 2.0)
    ry = max(1.0, (y1 - y0) / 2.0)
    for y in range(max(0, y0), min(len(pixels), y1 + 1)):
        for x in range(max(0, x0), min(len(pixels[0]), x1 + 1)):
            if math.pow((x - cx) / rx, 2) + math.pow((y - cy) / ry, 2) <= 1.0:
                pixels[y][x] = value


def create_mini_stage1(out_root: Path, *, num_images: int = 4, size: int = 160) -> Path:
    stage1_dir = out_root / "image"
    image_dir = stage1_dir / "images"
    mask_dir = stage1_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        img = _rgb_canvas(size, (24 + i * 12, 34, 48))
        mask = _gray_canvas(size)
        x0 = 28 + i * 10
        y0 = 34 + i * 7
        x1 = min(size - 12, x0 + 56)
        y1 = min(size - 12, y0 + 48)
        _draw_rect_rgb(img, (x0, y0, x1, y1), (220, 80 + i * 20, 70))
        _draw_rect_gray(mask, (x0, y0, x1, y1), 255)
        _write_png_rgb(image_dir / f"img{i:03d}.png", img)
        _write_png_gray(mask_dir / f"img{i:03d}.png", mask)

    return stage1_dir


def create_mini_stage2(
    out_root: Path,
    *,
    num_videos: int = 2,
    num_frames: int = 3,
    size: int = 160,
) -> Path:
    stage2_dir = out_root / "video"

    for v in range(num_videos):
        video_id = f"video{v + 1:03d}"
        frames_dir = stage2_dir / "JPEGImages" / video_id
        ann_dir = stage2_dir / "Annotations" / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        for t in range(num_frames):
            img = _rgb_canvas(size, (20, 38 + t * 16, 54 + v * 18))
            ann = _gray_canvas(size)
            x0 = 30 + t * 12
            y0 = 44 + v * 10
            x1 = min(size - 12, x0 + 50)
            y1 = min(size - 12, y0 + 42)
            _draw_ellipse_rgb(img, (x0, y0, x1, y1), (80, 210, 120))
            _draw_ellipse_gray(ann, (x0, y0, x1, y1), 1)
            _write_ppm(frames_dir / f"{t:05d}.jpg", img)
            _write_png_gray(ann_dir / f"{t:05d}.png", ann)

    return stage2_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        default="/tmp/efficienttam_test_data",
        help="Directory that will receive image/ and video/ mini datasets.",
    )
    parser.add_argument("--image-count", type=int, default=4)
    parser.add_argument("--video-count", type=int, default=2)
    parser.add_argument("--frame-count", type=int, default=3)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not remove --out-root before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    if not args.keep_existing:
        _reset_dir(out_root)
    else:
        out_root.mkdir(parents=True, exist_ok=True)

    image_root = create_mini_stage1(
        out_root,
        num_images=args.image_count,
        size=args.size,
    )
    video_root = create_mini_stage2(
        out_root,
        num_videos=args.video_count,
        num_frames=args.frame_count,
        size=args.size,
    )

    print("[prepare_mini_dataset] done")
    print(f"IMAGE_ROOT={image_root}")
    print(f"VIDEO_ROOT={video_root}")


if __name__ == "__main__":
    main()
