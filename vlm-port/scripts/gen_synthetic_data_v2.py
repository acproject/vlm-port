"""Synthetic multimodal data v2 — anti-shortcut recipe.

v1 (314 samples) let the projector collapse into a shortcut (mm loss variance
≈ 0 across samples, projector output cross-image cos 0.993): single-word
answers over tiny answer spaces (12 colors / 5 shapes / 5 counts / 16 words)
make the answer marginal distribution worth loss 9.11 with ZERO visual
information. Native Qwen on the same vision features scores ~98% semantic
accuracy → the features are fine, the data was the problem.

v2 recipe (3000 train + 200 val), output to data/images_v2/ + *_manifest_v2:
  1. caption (40%)  — full descriptive sentences templated from generation
                      params; blind guessing is expensive (long answers)
  2. joint QA (25%) — answers = f(two visual attributes); marginals can't
                      answer "3 red circles + 2 blue" conditional counts
  3. single QA (25%)— v1-style tasks, widened vocab (24 colors, 64 words),
                      80% short / 20% full-sentence answers
  4. either-or (10%)— "红色还是蓝色?" with random distractor → direct
                      visual binding, question text carries no answer info

Run:
    cd /home/acproject/usb/transplant/vlm-port
    ../venv-cu128/bin/python scripts/gen_synthetic_data_v2.py
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

random.seed(2026)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_DIR = DATA_DIR / "images_v2"
IMG_DIR.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

# 24 colors (v1's 12 kept with identical RGB for compatibility + 12 new)
COLORS = {
    "红色": (220, 40, 40), "深红": (150, 20, 30), "玫红": (220, 60, 120),
    "橙色": (240, 140, 50), "橘黄": (250, 170, 30), "棕色": (140, 90, 50),
    "黄色": (240, 210, 50), "土黄": (190, 160, 60),
    "绿色": (40, 180, 70), "草绿": (100, 200, 60), "墨绿": (20, 90, 50),
    "青色": (60, 200, 200),
    "蓝色": (40, 90, 220), "天蓝": (120, 180, 240), "深蓝": (20, 40, 120),
    "紫色": (150, 60, 200), "淡紫": (200, 170, 230),
    "粉色": (240, 150, 180),
    "黑色": (30, 30, 30), "白色": (240, 240, 240), "灰色": (140, 140, 140),
    "银色": (200, 200, 205), "卡其": (170, 160, 110), "藏青": (40, 50, 100),
}
SHAPES = ["圆形", "正方形", "三角形", "矩形", "五角星"]
OCR_WORDS = [
    # v1 16 (frozen — val set references them)
    "HELLO", "WORLD", "CAT", "DOG", "SUN", "MOON", "STAR", "TREE",
    "BOOK", "FISH", "RAIN", "SNOW", "2024", "8888", "ABCD", "LOVE",
    # 48 new: common words, number strings, mixed
    "APPLE", "HOUSE", "WATER", "LIGHT", "DREAM", "SMILE", "OCEAN", "MUSIC",
    "CLOUD", "RIVER", "STONE", "BIRD", "FLOWER", "GRAPE", "LEMON", "MANGO",
    "TIGER", "HORSE", "SHEEP", "MOUSE", "BREAD", "CAKE", "MILK", "TEA",
    "DOOR", "WALL", "ROAD", "PARK", "LAKE", "SNACK", "TOY", "GIFT",
    "2023", "2025", "1234", "5678", "0000", "1111", "3690", "1024",
    "AABB", "ZZZZ", "OK6", "AI26", "GPT4", "LLM9", "Q7V", "X1Y2",
]
COUNT_WORDS = {1: "一个", 2: "两个", 3: "三个", 4: "四个", 5: "五个"}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    raise RuntimeError("No TTF font found for OCR rendering")


def luminance(rgb) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def contrast_bg(rgb):
    return (40, 40, 40) if luminance(rgb) > 128 else (235, 235, 235)


def rand_size() -> int:
    return random.choices([224, 256, 288, 320], weights=[60, 25, 10, 5])[0]


def rand_color(exclude: set[str] = frozenset()) -> tuple[str, tuple]:
    while True:
        name = random.choice(list(COLORS))
        if name not in exclude:
            return name, COLORS[name]


# ── image makers (return image + param dict for Q/A templating) ──

def make_solid(size):
    name, rgb = rand_color()
    img = Image.new("RGB", (size, size), rgb)
    d = ImageDraw.Draw(img)
    for _ in range(30):  # subtle noise, not perfectly flat
        x, y = random.randint(0, size - 20), random.randint(0, size - 20)
        w = random.randint(4, 18)
        c = tuple(max(0, min(255, v + random.randint(-8, 8))) for v in rgb)
        d.ellipse([x, y, x + w, y + w], fill=c)
    return img, {"kind": "solid", "color": name}


def make_shape(size):
    shape = random.choice(SHAPES)
    cname, rgb = rand_color()
    bg = contrast_bg(rgb)
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    cx, cy = size // 2 + random.randint(-12, 12), size // 2 + random.randint(-12, 12)
    r = int(size * random.uniform(0.24, 0.38))
    if shape == "圆形":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == "正方形":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == "矩形":
        d.rectangle([cx - int(r * 1.3), cy - int(r * 0.6),
                     cx + int(r * 1.3), cy + int(r * 0.6)], fill=rgb)
    elif shape == "三角形":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=rgb)
    else:  # 五角星
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.4
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        d.polygon(pts, fill=rgb)
    bg_name = "深色" if luminance(bg) < 128 else "浅色"
    return img, {"kind": "shape", "shape": shape, "color": cname,
                 "bg_desc": bg_name}


def make_count(size, two_color: bool):
    """1-5 circles; optionally 2 colors for conditional counting."""
    n_total = random.randint(2, 5)
    if not two_color or n_total < 2:
        cname, rgb = rand_color()
        groups = [(cname, rgb, n_total)]
    else:
        n_a = random.randint(1, n_total - 1)
        c1, rgb1 = rand_color()
        c2, rgb2 = rand_color(exclude={c1})
        groups = [(c1, rgb1, n_a), (c2, rgb2, n_total - n_a)]
    img = Image.new("RGB", (size, size), (235, 235, 235))
    d = ImageDraw.Draw(img)
    r = random.randint(16, 22)
    placed = []
    circles = []
    for cname, rgb, n in groups:
        circles += [rgb] * n
    random.shuffle(circles)
    for rgb in circles:
        for _try in range(200):
            x = random.randint(r + 4, size - r - 4)
            y = random.randint(r + 4, size - r - 4)
            if all((x - px) ** 2 + (y - py) ** 2 > (2 * r + 8) ** 2 for px, py in placed):
                placed.append((x, y))
                break
        else:
            placed.append((x, y))
    for (x, y), rgb in zip(placed, circles):
        d.ellipse([x - r, y - r, x + r, y + r], fill=rgb)
    return img, {"kind": "count", "groups": groups}


def make_ocr(size):
    word = random.choice(OCR_WORDS)
    dark_text = random.random() < 0.5
    fg = (240, 240, 240) if dark_text else (20, 20, 20)
    bg_rgb = (20, 20, 20) if dark_text else (235, 235, 235)
    img = Image.new("RGB", (size, size), bg_rgb)
    d = ImageDraw.Draw(img)
    font = load_font(int(size * 0.28))
    bbox = d.textbbox((0, 0), word, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size - w) / 2 - bbox[0], (size - h) / 2 - bbox[1]), word,
           font=font, fill=fg)
    return img, {"kind": "ocr", "word": word,
                 "bg_desc": "黑色" if dark_text else "白色"}


# ── Q/A generators ──

Q_CAPTION = ["描述一下这张图片。", "这张图片里有什么？", "请简要描述图片内容。",
             "说说你看到了什么。"]
Q_COLOR = ["这张图片的主导颜色是什么？", "图片主要是什么颜色？", "这张图是什么颜色的？"]
Q_SHAPE = ["图中画的是什么形状？", "这张图片里是什么图形？"]
Q_COUNT = ["图中有几个圆形？", "数一数图里有几个圆？"]
Q_OCR = ["图片中写了什么文字？", "图里的文字是什么？"]


def maybe_full(ans_short: str, full: str, p_full: float = 0.2) -> str:
    return full if random.random() < p_full else ans_short


def gen_caption(params) -> tuple[str, str]:
    q = random.choice(Q_CAPTION)
    k = params["kind"]
    if k == "solid":
        c = params["color"]
        a = random.choice([
            f"这是一张以{c}为主色调的图片，带有轻微的明暗变化。",
            f"这张图片整体呈现{c}，画面简单均匀。",
            f"图片通篇是{c}色，没有其他明显内容。",
        ])
    elif k == "shape":
        a = random.choice([
            f"图片的背景是{params['bg_desc']}的，中央有一个{params['color']}的{params['shape']}。",
            f"图中是一个{params['color']}的{params['shape']}，放在{params['bg_desc']}背景上。",
            f"画面{params['bg_desc']}背景的中央画着一个{params['color']}{params['shape']}。",
        ])
    elif k == "count":
        groups = params["groups"]
        if len(groups) == 1:
            c, _, n = groups[0]
            a = random.choice([
                f"图中有{COUNT_WORDS[n]}{c}的圆形，背景是浅灰色。",
                f"画面上散布着{COUNT_WORDS[n]}{c}圆，底色为浅灰。",
            ])
        else:
            (c1, _, n1), (c2, _, n2) = groups
            a = random.choice([
                f"图中有{COUNT_WORDS[n1]}{c1}圆形和{COUNT_WORDS[n2]}{c2}圆形，背景是浅灰色。",
                f"画面里有{COUNT_WORDS[n1]}{c1}的圆，还有{COUNT_WORDS[n2]}{c2}的圆。",
            ])
    else:  # ocr
        w, bg = params["word"], params["bg_desc"]
        a = random.choice([
            f'图片中央写着一行文字"{w}"，背景是{bg}的。',
            f'图中是{bg}背景，中间写着"{w}"。',
            f'画面{bg}底色上有一行字：{w}。',
        ])
    return q, a


def gen_joint(params) -> tuple[str, str]:
    k = params["kind"]
    if k == "shape":
        s, c = params["shape"], params["color"]
        return random.choice([
            (f"图中画了一个什么颜色的什么形状？", f"一个{c}的{s}。"),
            (f"图中的形状是什么？是什么颜色？", f"{s}，{c}。"),
            (f"这个图形的形状和颜色分别是什么？", f"形状是{s}，颜色是{c}。"),
        ])
    if k == "count" and len(params["groups"]) == 2:
        (c1, _, n1), (c2, _, n2) = params["groups"]
        # conditional counting: answer depends on WHICH color is asked
        if random.random() < 0.5:
            q = f"图中有几个{c1}圆形？"
            return q, maybe_full(COUNT_WORDS[n1], f"有{COUNT_WORDS[n1]}。")
        q = f"图中有几个{c2}圆形？"
        return q, maybe_full(COUNT_WORDS[n2], f"有{COUNT_WORDS[n2]}。")
    if k == "count":  # single color — count + color joint
        c, _, n = params["groups"][0]
        return ("图中有几个圆形？它们是什么颜色的？",
                f"{COUNT_WORDS[n]}，都是{c}的。")
    if k == "ocr":
        w, bg = params["word"], params["bg_desc"]
        return random.choice([
            (f"图里写的是什么文字？背景是什么颜色？", f'"{w}"，{bg}背景。'),
            (f"图片中的文字和背景颜色分别是什么？", f"文字是{w}，背景是{bg}的。"),
        ])
    # solid → no second attribute; degrade to single QA
    return gen_single(params)


def gen_single(params, force_short: bool = False) -> tuple[str, str]:
    k = params["kind"]
    p = 0.0 if force_short else 0.2
    if k == "solid":
        c = params["color"]
        return random.choice(Q_COLOR), maybe_full(c, f"这张图的主色调是{c}。", p)
    if k == "shape":
        s, c = params["shape"], params["color"]
        return random.choice(Q_SHAPE), maybe_full(s, f"图中画的是一个{c}的{s}。", p)
    if k == "count":
        c, _, n = params["groups"][0]
        return random.choice(Q_COUNT), maybe_full(COUNT_WORDS[n],
                                                  f"图中有{COUNT_WORDS[n]}圆形。", p)
    return random.choice(Q_OCR), maybe_full(
        params["word"], f'图里写的文字是"{params["word"]}"。', p)


def gen_either(params) -> tuple[str, str]:
    k = params["kind"]
    if k == "solid":
        c = params["color"]
        d, _ = rand_color(exclude={c})
        return (f"这张图片的主色调是{c}还是{d}？", f"{c}。")
    if k == "shape":
        s, c = params["shape"], params["color"]
        d, _ = rand_color(exclude={c})
        return (f"图中的{s}是{c}还是{d}的？", f"是{c}的。")
    if k == "count":
        c, _, n = params["groups"][0]
        d, _ = rand_color(exclude={c})
        return (f"图中的圆形都是{c}的还是{d}的？", f"都是{c}的。")
    w = params["word"]
    d = random.choice([x for x in OCR_WORDS if x != w])
    return (f'图里写的是"{w}"还是"{d}"？', f'"{w}"。')


def make_scene(task: str):
    """Pick a scene kind appropriate for the task, render image + params."""
    if task == "caption":
        k = random.choices(["solid", "shape", "count", "ocr"],
                           weights=[15, 30, 35, 20])[0]
    elif task == "joint":
        k = random.choices(["shape", "count", "ocr"], weights=[40, 30, 30])[0]
    elif task == "either":
        k = random.choices(["solid", "shape", "count", "ocr"],
                           weights=[30, 40, 10, 20])[0]
    else:  # single
        k = random.choices(["solid", "shape", "count", "ocr"],
                           weights=[25, 25, 25, 25])[0]
    size = rand_size()
    if k == "solid":
        return make_solid(size)
    if k == "shape":
        return make_shape(size)
    if k == "count":
        two_color = (task == "joint") and random.random() < 0.7
        return make_count(size, two_color)
    return make_ocr(size)


def main():
    quotas = {
        "train": {"caption": 1200, "joint": 750, "single": 750, "either": 300},
        "val": {"caption": 80, "joint": 50, "single": 50, "either": 20},
    }
    records = {"train": [], "val": []}
    task_of = {"train": [], "val": []}  # parallel task labels for stats
    idx = 0
    for split, quota in quotas.items():
        for task, n in quota.items():
            for _ in range(n):
                img, params = make_scene(task)
                if task == "caption":
                    q, a = gen_caption(params)
                elif task == "joint":
                    q, a = gen_joint(params)
                elif task == "either":
                    q, a = gen_either(params)
                else:
                    q, a = gen_single(params, force_short=(split == "val"))
                idx += 1
                name = f"img_{idx:04d}.png"
                img.save(IMG_DIR / name)
                records[split].append(
                    {"image": f"images_v2/{name}", "question": q, "answer": a})
                task_of[split].append(task)
        random.shuffle(records[split])
        task_of[split] = list(zip(records[split], task_of[split]))

    for split in ("train", "val"):
        with open(DATA_DIR / f"{split}_manifest_v2.jsonl", "w") as f:
            for r in records[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for split in ("train", "val"):
        lens = [len(r["answer"]) for r in records[split]]
        print(f"{split}: {len(records[split])} samples, "
              f"answer len mean={sum(lens)/len(lens):.1f} max={max(lens)}")
    print(f"Images: {idx} → {IMG_DIR}")
    print("Train task mix:", Counter(t for _, t in task_of["train"]))
    print("\nSample records:")
    for r, t in random.sample(task_of["train"], 8):
        print(f"  [{t:7s}] q='{r['question']}' a='{r['answer']}'")


if __name__ == "__main__":
    main()
