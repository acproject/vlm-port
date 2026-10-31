"""Generate synthetic multimodal training data for Stage 1 alignment.

Tasks (unambiguous ground truth, fully offline):
  1. Color  — solid color image → "这张图片的主导颜色是什么？" → "绿色"
  2. Shape  — colored shape on contrasting bg → "图中画的是什么形状？" → "圆形"
  3. Count  — 1-5 circles → "图中有几个圆形？" → "三个"
  4. OCR    — large text → "图片中写了什么文字？" → "HELLO"

Output: data/images/*.png + data/train_manifest.jsonl + data/val_manifest.jsonl

Run:
    cd /home/acproject/usb/transplant/vlm-port
    ../venv-cu128/bin/python scripts/gen_synthetic_data.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

COLORS = {
    "红色": (220, 40, 40), "绿色": (40, 180, 70), "蓝色": (40, 90, 220),
    "黄色": (240, 210, 50), "紫色": (150, 60, 200), "橙色": (240, 140, 50),
    "粉色": (240, 150, 180), "棕色": (140, 90, 50), "黑色": (30, 30, 30),
    "白色": (240, 240, 240), "灰色": (140, 140, 140), "青色": (60, 200, 200),
}
SHAPES = ["圆形", "正方形", "三角形", "矩形", "五角星"]
OCR_WORDS = [
    "HELLO", "WORLD", "CAT", "DOG", "SUN", "MOON", "STAR", "TREE",
    "BOOK", "FISH", "RAIN", "SNOW", "2024", "8888", "ABCD", "LOVE",
]
COUNT_WORDS = {1: "一个", 2: "两个", 3: "三个", 4: "四个", 5: "五个"}

Q_COLOR = ["这张图片的主导颜色是什么？", "图片主要是什么颜色？", "这张图是什么颜色的？"]
Q_SHAPE = ["图中画的是什么形状？", "这张图片里是什么图形？"]
Q_COUNT = ["图中有几个圆形？", "数一数图里有几个圆？"]
Q_OCR = ["图片中写了什么文字？", "图里的文字是什么？"]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    raise RuntimeError("No TTF font found for OCR rendering")


def luminance(rgb) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def contrast_bg(rgb):
    """Dark bg for light shapes, light bg for dark shapes."""
    return (40, 40, 40) if luminance(rgb) > 128 else (235, 235, 235)


def rand_size() -> int:
    return random.choice([224, 224, 224, 256])  # mostly 224 → 49 visual tokens


def make_color_img(color_rgb, size):
    img = Image.new("RGB", (size, size), color_rgb)
    # subtle gradient noise so it's not perfectly flat
    d = ImageDraw.Draw(img)
    for _ in range(30):
        x, y = random.randint(0, size - 20), random.randint(0, size - 20)
        w = random.randint(4, 18)
        c = tuple(max(0, min(255, v + random.randint(-8, 8))) for v in color_rgb)
        d.ellipse([x, y, x + w, y + w], fill=c)
    return img


def make_shape_img(shape, color_rgb, size):
    bg = contrast_bg(color_rgb)
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    cx, cy = size // 2 + random.randint(-10, 10), size // 2 + random.randint(-10, 10)
    r = int(size * random.uniform(0.28, 0.36))
    if shape == "圆形":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color_rgb)
    elif shape == "正方形":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color_rgb)
    elif shape == "矩形":
        d.rectangle([cx - int(r * 1.3), cy - int(r * 0.6), cx + int(r * 1.3), cy + int(r * 0.6)], fill=color_rgb)
    elif shape == "三角形":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color_rgb)
    elif shape == "五角星":
        import math
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.4
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        d.polygon(pts, fill=color_rgb)
    return img


def make_count_img(n, color_rgb, size):
    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)
    r = random.randint(16, 22)
    placed = []
    for _ in range(n):
        for _try in range(200):
            x = random.randint(r + 4, size - r - 4)
            y = random.randint(r + 4, size - r - 4)
            if all((x - px) ** 2 + (y - py) ** 2 > (2 * r + 8) ** 2 for px, py in placed):
                placed.append((x, y))
                break
        else:
            placed.append((x, y))
    for x, y in placed:
        d.ellipse([x - r, y - r, x + r, y + r], fill=color_rgb)
    return img


def make_ocr_img(word, size):
    img = Image.new("RGB", (size, size), (20, 20, 20))
    d = ImageDraw.Draw(img)
    font = load_font(int(size * 0.28))
    bbox = d.textbbox((0, 0), word, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size - w) / 2 - bbox[0], (size - h) / 2 - bbox[1]), word,
           font=font, fill=(240, 240, 240))
    return img


def main():
    train, val = [], []
    idx = 0

    def add(img, question, answer, split):
        nonlocal idx
        idx += 1
        name = f"img_{idx:04d}.png"
        img.save(IMG_DIR / name)
        rec = {"image": f"images/{name}", "question": question, "answer": answer}
        (train if split == "train" else val).append(rec)

    # 1) Colors: 12 colors × 9 train + 1 val each
    for cname, rgb in COLORS.items():
        for k in range(10):
            split = "val" if k == 9 else "train"
            add(make_color_img(rgb, rand_size()), random.choice(Q_COLOR), cname, split)

    # 2) Shapes: 5 shapes × 12 train + 2 val each
    for shape in SHAPES:
        for k in range(14):
            split = "val" if k >= 12 else "train"
            rgb = random.choice(list(COLORS.values()))
            add(make_shape_img(shape, rgb, rand_size()), random.choice(Q_SHAPE), shape, split)

    # 3) Counting: 5 counts × 10 train + 2 val each
    for n in range(1, 6):
        for k in range(12):
            split = "val" if k >= 10 else "train"
            rgb = random.choice(list(COLORS.values()))
            add(make_count_img(n, rgb, rand_size()), random.choice(Q_COUNT), COUNT_WORDS[n], split)

    # 4) OCR: 16 words × 6 train + 1 val each
    for word in OCR_WORDS:
        for k in range(7):
            split = "val" if k == 6 else "train"
            add(make_ocr_img(word, rand_size()), random.choice(Q_OCR), word, split)

    random.shuffle(train)
    random.shuffle(val)

    with open(DATA_DIR / "train_manifest.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(DATA_DIR / "val_manifest.jsonl", "w") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Train: {len(train)} samples, Val: {len(val)} samples")
    print(f"Images: {idx} → {IMG_DIR}")
    from collections import Counter
    print("Train task mix:", Counter(
        "color" if q in Q_COLOR else "shape" if q in Q_SHAPE
        else "count" if q in Q_COUNT else "ocr"
        for q in (r["question"] for r in train)))


if __name__ == "__main__":
    main()
