from dotenv import load_dotenv
load_dotenv()

import os
import io
import time
import base64
import json
import math
import sqlite3
import secrets
import threading
import traceback
import requests
import numpy as np
import cv2
import cloudinary
import cloudinary.uploader
from datetime import datetime, timedelta
from flask import (Flask, request, jsonify, render_template, session,
                   redirect, url_for, send_from_directory, make_response)
from flask_cors import CORS
from flask_compress import Compress
from PIL import Image, ImageDraw, ImageFont
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "textora_stable_secret_key_2024_do_not_change")
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=30)
app.config["SESSION_COOKIE_SECURE"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 31536000
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
CORS(app, supports_credentials=True)
Compress(app)

cloudinary.config(
    cloud_name=(os.environ.get("CLOUDINARY_CLOUD_NAME") or "").strip(),
    api_key=(os.environ.get("CLOUDINARY_API_KEY") or "").strip(),
    api_secret=(os.environ.get("CLOUDINARY_API_SECRET") or "").strip(),
)

DB_PATH = os.path.join(os.path.dirname(__file__), "textora.db")

FONT_PATHS = {
    "bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "regular": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "bold-oblique": "/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",
    "oblique": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
}

EXTRA_FONT_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "/usr/share/fonts/truetype",
]

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")


def download_font_dynamic(family_name, bold=False):
    """
    Download a single font from Google Fonts API at runtime and cache in FONTS_DIR.
    Uses GOOGLE_API_KEY env var to query the fonts directory.
    Returns local TTF path on success, None on failure.
    """
    import re as _re
    clean = _re.sub(r"[^a-z0-9]", "_", family_name.lower())
    weight = "bold" if bold else "regular"
    cache_path = os.path.join(FONTS_DIR, f"gf_{clean}_{weight}.ttf")
    if os.path.exists(cache_path):
        return cache_path
    try:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        resp = requests.get(
            "https://www.googleapis.com/webfonts/v1/webfonts",
            params={"key": api_key, "family": family_name},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if not items:
            raise ValueError(f"Font '{family_name}' not found in Google Fonts")
        files = items[0].get("files", {})
        url = files.get("700" if bold else "regular") or files.get("regular") or next(iter(files.values()), None)
        if not url:
            raise ValueError(f"No file URL for '{family_name}'")
        ttf = requests.get(url, timeout=30)
        ttf.raise_for_status()
        os.makedirs(FONTS_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            f.write(ttf.content)
        print(f"[FONT-DL] Cached {family_name} ({weight}): {cache_path}")
        return cache_path
    except Exception as ex:
        print(f"[FONT-DL] Failed '{family_name}': {ex}")
        return None


def download_fonts():
    """Download a set of design-ready Google Fonts on startup (background thread)."""
    font_urls = {
        "Anton-Regular":    "https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
        "BebasNeue-Regular":"https://github.com/google/fonts/raw/main/ofl/bebasneue/BebasNeue-Regular.ttf",
        "Lato-Bold":        "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Bold.ttf",
        "Lato-Regular":     "https://github.com/google/fonts/raw/main/ofl/lato/Lato-Regular.ttf",
        "Ubuntu-Bold":      "https://github.com/google/fonts/raw/main/ufl/ubuntu/Ubuntu-Bold.ttf",
        "Ubuntu-Regular":   "https://github.com/google/fonts/raw/main/ufl/ubuntu/Ubuntu-Regular.ttf",
        "Arvo-Bold":        "https://github.com/google/fonts/raw/main/ofl/arvo/Arvo-Bold.ttf",
        "Arvo-Regular":     "https://github.com/google/fonts/raw/main/ofl/arvo/Arvo-Regular.ttf",
        "PTSans-Bold":      "https://github.com/google/fonts/raw/main/ofl/ptsans/PT_Sans-Web-Bold.ttf",
        "Tinos-Bold":       "https://github.com/google/fonts/raw/main/apache/tinos/Tinos-Bold.ttf",
        "Tinos-Regular":    "https://github.com/google/fonts/raw/main/apache/tinos/Tinos-Regular.ttf",
    }
    os.makedirs(FONTS_DIR, exist_ok=True)
    for name, url in font_urls.items():
        path = os.path.join(FONTS_DIR, f"{name}.ttf")
        if not os.path.exists(path):
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200:
                    with open(path, "wb") as f:
                        f.write(r.content)
                    print(f"[FONTS] Downloaded: {name}")
                else:
                    print(f"[FONTS] HTTP {r.status_code} for {name}")
            except Exception as e:
                print(f"[FONTS] Could not download {name}: {e}")


def get_best_font(font_name, font_size, bold=False):
    """Return the best matching PIL font for a given name/size/weight."""
    font_map = {
        "lato":         ("Lato-Bold.ttf"         if bold else "Lato-Regular.ttf"),
        "ubuntu":       ("Ubuntu-Bold.ttf"        if bold else "Ubuntu-Regular.ttf"),
        "arvo":         ("Arvo-Bold.ttf"          if bold else "Arvo-Regular.ttf"),
        "tinos":        ("Tinos-Bold.ttf"         if bold else "Tinos-Regular.ttf"),
        "ptsans":       "PTSans-Bold.ttf",
        "pt":           "PTSans-Bold.ttf",
        "anton":        "Anton-Regular.ttf",
        "bebas":        "BebasNeue-Regular.ttf",
        "bebasneue":    "BebasNeue-Regular.ttf",
        # Aliases for fonts not downloaded — map to closest available
        "montserrat":   ("Lato-Bold.ttf"         if bold else "Lato-Regular.ttf"),
        "roboto":       ("Ubuntu-Bold.ttf"        if bold else "Ubuntu-Regular.ttf"),
        "opensans":     ("Lato-Bold.ttf"          if bold else "Lato-Regular.ttf"),
        "oswald":       "Anton-Regular.ttf",
        "raleway":      ("Ubuntu-Bold.ttf"        if bold else "Ubuntu-Regular.ttf"),
        "inter":        ("Lato-Bold.ttf"          if bold else "Lato-Regular.ttf"),
        "playfair":     ("Arvo-Bold.ttf"          if bold else "Arvo-Regular.ttf"),
        "merriweather": ("Tinos-Bold.ttf"         if bold else "Tinos-Regular.ttf"),
    }
    key = (font_name or "lato").lower().replace(" ", "").replace("-", "")
    filename = font_map.get(key)
    font_path = None
    if filename:
        candidate = os.path.join(FONTS_DIR, filename)
        if os.path.exists(candidate):
            font_path = candidate
    if not font_path:
        for fname in font_map.values():
            candidate = os.path.join(FONTS_DIR, fname)
            if os.path.exists(candidate):
                font_path = candidate
                break
    if not font_path:
        fallback = FONT_PATHS["bold"] if bold else FONT_PATHS["regular"]
        if os.path.exists(fallback):
            font_path = fallback
    try:
        if font_path:
            return ImageFont.truetype(font_path, font_size)
    except Exception:
        pass
    return ImageFont.load_default()


def _weighted_strip(arr):
    """
    Collapse a (N, M, 3) strip to (M, 3) using exponential-decay weighting.
    Index 0 is the INNERMOST row/col — it gets the highest weight.
    This prevents far-away pixels (design borders, nearby text)
    from distorting the boundary colour estimate.
    """
    n = arr.shape[0]
    if n == 1:
        return arr[0].astype(np.float32)
    # exponential decay: row 0 weight = 1, row n-1 weight = 0.5^(n-1)
    weights = np.array([0.5 ** i for i in range(n)], dtype=np.float32)
    weights /= weights.sum()
    return (arr.astype(np.float32) * weights.reshape(n, 1, 1)).sum(axis=0)


def _reconstruct_bg_coons(img_arr, x1, y1, x2, y2, strip=8):
    """
    Reconstruct the background inside a bounding box using a Coons-patch
    bilinear interpolation from the four surrounding pixel strips.

    Improvements over naive averaging:
    - Exponential-decay weighting: the pixel immediately adjacent to the bbox
      gets the highest weight; pixels further away are down-weighted.
      This prevents design elements (borders, nearby text) from poisoning the
      boundary estimate.
    - Edge feathering: a smooth cosine blend at the bbox border ensures the
      reconstructed region integrates seamlessly with the original image even
      on complex non-linear gradients.

    Algorithm (Coons patch):
        bg(tx, ty) = (1-ty)*top(tx) + ty*bot(tx)
                   + (1-tx)*lft(ty) + tx*rgt(ty)
                   - corner_bilinear(tx, ty)
    """
    h, w = img_arr.shape[:2]
    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(w, x2); y2c = min(h, y2)
    rh = y2c - y1c; rw = x2c - x1c
    if rh <= 0 or rw <= 0:
        return img_arr.copy()

    samp = max(2, min(strip, rh // 2, rw // 2))

    # ── per-column top / bottom boundary values (innermost pixel weighted most) ──
    if y1c >= samp:
        # rows immediately OUTSIDE the top edge, innermost first
        raw_top = img_arr[y1c - samp:y1c, x1c:x2c][::-1]  # flip so row0=innermost
        top_vals = _weighted_strip(raw_top)
    else:
        top_vals = img_arr[y1c:y1c + 1, x1c:x2c].astype(np.float32)[0]

    if y2c + samp <= h:
        raw_bot = img_arr[y2c:y2c + samp, x1c:x2c]  # row0 already innermost
        bot_vals = _weighted_strip(raw_bot)
    else:
        bot_vals = img_arr[y2c - 1:y2c, x1c:x2c].astype(np.float32)[0]

    # ── per-row left / right boundary values ─────────────────────────────────
    if x1c >= samp:
        raw_lft = img_arr[y1c:y2c, x1c - samp:x1c].transpose(1, 0, 2)[::-1]
        lft_vals = _weighted_strip(raw_lft)   # shape (rh, 3)
    else:
        lft_vals = img_arr[y1c:y2c, x1c:x1c + 1].astype(np.float32)[:, 0, :]

    if x2c + samp <= w:
        raw_rgt = img_arr[y1c:y2c, x2c:x2c + samp].transpose(1, 0, 2)
        rgt_vals = _weighted_strip(raw_rgt)   # shape (rh, 3)
    else:
        rgt_vals = img_arr[y1c:y2c, x2c - 1:x2c].astype(np.float32)[:, 0, :]

    # ── Coons-patch blend ─────────────────────────────────────────────────────
    ty = np.linspace(0.0, 1.0, rh, dtype=np.float32).reshape(rh, 1, 1)
    tx = np.linspace(0.0, 1.0, rw, dtype=np.float32).reshape(1, rw, 1)

    horiz = (1 - ty) * top_vals.reshape(1, rw, 3) + ty * bot_vals.reshape(1, rw, 3)
    vert  = (1 - tx) * lft_vals.reshape(rh, 1, 3) + tx * rgt_vals.reshape(rh, 1, 3)

    tl = top_vals[0].reshape(1, 1, 3);  tr = top_vals[-1].reshape(1, 1, 3)
    bl = bot_vals[0].reshape(1, 1, 3);  br = bot_vals[-1].reshape(1, 1, 3)
    corner = (1-ty)*(1-tx)*tl + (1-ty)*tx*tr + ty*(1-tx)*bl + ty*tx*br

    bg = np.clip(horiz + vert - corner, 0, 255).astype(np.float32)

    # ── Edge feathering — blend Coons result with original at the bbox boundary ──
    # A cosine ramp across `feather` pixels fades from original→Coons smoothly,
    # eliminating any hard rectangular seam even on complex nonlinear gradients.
    feather = max(1, min(4, rh // 6, rw // 6))
    alpha = np.ones((rh, rw), dtype=np.float32)
    for k in range(feather):
        t = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / (feather + 1))   # 0→1
        alpha[k,       :] = np.minimum(alpha[k,       :], t)
        alpha[-(k+1),  :] = np.minimum(alpha[-(k+1),  :], t)
        alpha[:,       k] = np.minimum(alpha[:,       k], t)
        alpha[:, -(k+1)] = np.minimum(alpha[:, -(k+1)], t)
    alpha = alpha.reshape(rh, rw, 1)

    orig_region = img_arr[y1c:y2c, x1c:x2c].astype(np.float32)
    blended = alpha * bg + (1.0 - alpha) * orig_region

    result = img_arr.copy()
    result[y1c:y2c, x1c:x2c] = np.clip(blended, 0, 255).astype(np.uint8)
    print(f"[BG] Coons-patch fill ({rh}×{rw}px, strip={samp}, feather={feather})")
    return result


def _reconstruct_bg_multipass(img_arr, x1, y1, x2, y2, strip=8, passes=4):
    """
    Multi-pass Coons-patch fill for large regions.

    Large regions cannot be accurately filled with a single bilinear patch
    because a complex gradient cannot be represented as four straight boundary
    curves.  Instead, we apply the Coons patch iteratively:

      Pass 1 — full bbox, sampling from original pixels outside  →  rough fill
      Pass 2 — shrunk bbox (strip inward on each side), boundary is now the
               already-filled outer ring  →  better approximation for interior
      …
      Pass N — center area refined by all previous rings

    Each inner pass uses the progressively more accurate outer ring as its
    boundary, building toward a natural-looking gradient even on complex
    multi-color backgrounds.
    """
    arr = img_arr.copy()
    h, w = arr.shape[:2]
    x1c = max(0, x1); y1c = max(0, y1)
    x2c = min(w, x2); y2c = min(h, y2)

    shrink_per_pass = max(4, strip)
    for p in range(passes):
        s = p * shrink_per_pass
        bx1 = x1c + s; by1 = y1c + s
        bx2 = x2c - s; by2 = y2c - s
        if bx2 <= bx1 + 2 or by2 <= by1 + 2:
            break
        arr = _reconstruct_bg_coons(arr, bx1, by1, bx2, by2, strip=strip)
    return arr


def reconstruct_background(img, x1, y1, x2, y2):
    """
    Gradient-aware background reconstruction.
    Automatically selects single-pass (small) or multi-pass (large) Coons patch.
    """
    img_arr = np.array(img, dtype=np.uint8)
    rh = max(0, y2 - y1)
    rw = max(0, x2 - x1)
    if rh > 50 or rw > 150:
        result = _reconstruct_bg_multipass(img_arr, x1, y1, x2, y2)
    else:
        result = _reconstruct_bg_coons(img_arr, x1, y1, x2, y2)
    return Image.fromarray(result)


def reconstruct_background_clean(img, x1, y1, x2, y2):
    """Alias for reconstruct_background — used for erase-only edits."""
    return reconstruct_background(img, x1, y1, x2, y2)


def detect_text_color(img, x, y, width, height):
    """
    Sample pixels inside the bounding box and find the dominant text color
    (the color most different from the local background).
    """
    try:
        pad = 4
        x1 = max(0, x + pad)
        y1 = max(0, y + pad)
        x2 = min(img.width,  x + width  - pad)
        y2 = min(img.height, y + height - pad)
        if x2 <= x1 or y2 <= y1:
            return "#ffffff"

        region = np.array(img.crop((x1, y1, x2, y2)), dtype=np.float32)
        pixels = region.reshape(-1, 3)

        bg_strip = np.array(img)[max(0, y - 4):y, x:x + width]
        if bg_strip.size > 0:
            bg_color = bg_strip.mean(axis=(0, 1))
        else:
            bg_color = np.array([128.0, 128.0, 128.0])

        distances  = np.abs(pixels - bg_color).sum(axis=1)
        text_pixels = pixels[distances > 60]
        if len(text_pixels) == 0:
            text_pixels = pixels

        avg = text_pixels.mean(axis=0).astype(int)
        return "#{:02x}{:02x}{:02x}".format(int(avg[0]), int(avg[1]), int(avg[2]))
    except Exception:
        return "#ffffff"


def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hashed TEXT NOT NULL,
            avatar_url TEXT DEFAULT '',
            plan_type TEXT DEFAULT 'free',
            bio TEXT DEFAULT '',
            website TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            last_login TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            project_name TEXT DEFAULT 'Untitled Project',
            original_image_url TEXT DEFAULT '',
            edited_image_url TEXT DEFAULT '',
            width INTEGER DEFAULT 0,
            height INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            last_edited_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            settings_json TEXT DEFAULT '{}',
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()
    print("[DB] Database initialized.")


init_db()
threading.Thread(target=download_fonts, daemon=True).start()


def prepare_image_for_ocr(image_bytes):
    """Resize images wider than 1500px before sending to OCR.space to stay under 1 MB and process faster."""
    img = Image.open(io.BytesIO(image_bytes))
    orig_width, orig_height = img.size
    if orig_width > 1500:
        ratio = 1500 / orig_width
        new_height = int(orig_height * ratio)
        img = img.resize((1500, new_height), Image.LANCZOS)
        print(f"[OCR-PREP] Resized {orig_width}x{orig_height} → 1500x{new_height} for OCR")
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    out.seek(0)
    return out.getvalue(), orig_width, orig_height, img.width, img.height


def replace_text_fast(image_bytes, x, y, width, height, new_text, font_size,
                      text_color, bold, italic=False, font_family="",
                      detected_color=None, alignment="center", letter_spacing=0,
                      orig_x=None, orig_y=None, orig_w=None, orig_h=None):
    """
    Text replacement using gradient-aware background reconstruction.
    Supports two-position mode: remove old text from orig position,
    render new text at (x, y). Falls back gracefully everywhere.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    removing = not new_text or not new_text.strip()
    padding = 6   # 6px covers anti-aliased font edges on all sides

    # Resolve original position (defaults to current position if not dragged)
    ox = orig_x if orig_x is not None else x
    oy = orig_y if orig_y is not None else y
    ow = orig_w if orig_w is not None else width
    oh = orig_h if orig_h is not None else height

    # Step 1 — Always erase the ORIGINAL position first
    ox1 = max(0, ox - padding)
    oy1 = max(0, oy - padding)
    ox2 = min(img.width,  ox + ow + padding)
    oy2 = min(img.height, oy + oh + padding)
    img = reconstruct_background(img, ox1, oy1, ox2, oy2)

    # Step 2 — If the box was dragged to a different spot, also clear new area
    position_moved = abs(x - ox) > 5 or abs(y - oy) > 5
    if position_moved and not removing:
        nx1 = max(0, x - padding)
        ny1 = max(0, y - padding)
        nx2 = min(img.width,  x + width  + padding)
        ny2 = min(img.height, y + height + padding)
        img = reconstruct_background(img, nx1, ny1, nx2, ny2)

    # Step 3 — If removing (no text), return the already-reconstructed image
    if removing:
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()

    # — Auto font size: 75 % of bbox height as default —
    auto_size = max(8, int(height * 0.75))
    final_size = max(8, min(200, int(font_size) if font_size else auto_size))

    # — Auto color: use detected_color unless caller explicitly overrode —
    final_color = text_color if (text_color and text_color not in ("#000000", "")) \
                  else (detected_color or "#ffffff")

    # — Font: prefer downloaded Google Font, fall back to system DejaVu —
    font_name_clean = (font_family or "lato").lower().replace(" ", "").replace("-", "")
    if font_name_clean in ("", "auto", "autodetectmatch", "dejavu", "dejavusans"):
        font_name_clean = "lato"
    font = get_best_font(font_name_clean, final_size, bold=bold)

    # — Measure text for alignment —
    draw = ImageDraw.Draw(img)
    try:
        bbox_m = draw.textbbox((0, 0), new_text, font=font)
        text_width = bbox_m[2] - bbox_m[0]
        text_height = bbox_m[3] - bbox_m[1]
    except Exception:
        text_width = len(new_text) * (final_size // 2)
        text_height = final_size

    # — Compute draw X based on alignment —
    if alignment == "center":
        draw_x = x + max(0, (width - text_width) // 2)
    elif alignment == "right":
        draw_x = x + max(0, width - text_width)
    else:  # left (default)
        draw_x = x
    draw_x = max(0, min(draw_x, img.width - 1))

    # — Compute draw Y (vertically centered in bbox) —
    draw_y = y + max(0, (height - text_height) // 2)
    draw_y = max(0, min(draw_y, img.height - 1))

    # — Draw text —
    tc = final_color.lstrip("#")
    if len(tc) == 3:
        tc = tc[0]*2 + tc[1]*2 + tc[2]*2
    try:
        fill_rgb = (int(tc[0:2], 16), int(tc[2:4], 16), int(tc[4:6], 16))
    except Exception:
        fill_rgb = (255, 255, 255)

    draw.text((draw_x, draw_y), new_text, font=font, fill=fill_rgb)

    output = io.BytesIO()
    img.save(output, format="PNG", optimize=True)
    output.seek(0)
    return output.getvalue()


def _parse_ocr_result(result, region_id=1):
    """Extract text regions from an OCR.space JSON response."""
    text_regions = []
    if result.get("ParsedResults"):
        for parsed in result["ParsedResults"]:
            overlay = parsed.get("TextOverlay", {})
            lines = overlay.get("Lines", [])
            for line in lines:
                words = line.get("Words", [])
                if not words:
                    continue
                line_text = " ".join([w["WordText"] for w in words])
                min_left  = min(w["Left"]               for w in words)
                min_top   = min(w["Top"]                for w in words)
                max_right = max(w["Left"] + w["Width"]  for w in words)
                max_bot   = max(w["Top"]  + w["Height"] for w in words)
                text_regions.append({
                    "id": region_id,
                    "text": line_text,
                    "x": int(min_left),
                    "y": int(min_top),
                    "width": int(max_right - min_left),
                    "height": int(max_bot  - min_top),
                    "type": "straight",
                    "confidence": 95.0,
                })
                region_id += 1
    return text_regions


def _ocrspace_call(payload, label, timeout_s):
    """Single OCR.space API call; returns parsed regions or raises."""
    t0 = time.time()
    print(f"[OCR] {label} (timeout={timeout_s}s)…")
    response = requests.post(
        "https://api.ocr.space/parse/image",
        data=payload,
        timeout=timeout_s,
    )
    result = response.json()
    elapsed = time.time() - t0
    print(f"[OCR] {label} — {elapsed:.1f}s")
    return _parse_ocr_result(result)


def detect_text_ocrspace(image_bytes, image_url=None):
    """
    Call OCR.space with automatic retry + engine fallback.

    Strategy (fastest → slowest):
      Round 1 — URL-based (if image_url provided): Engine 2 then Engine 1, 12 s each
                 Sending a URL means OCR.space fetches the image itself — far
                 less data on the wire and avoids base64 throttling.
      Round 2 — base64 fallback: Engine 1, 25 s
                 Only used when no URL is available.
    """
    api_key = os.environ.get("OCRSPACE_API_KEY", "")
    base_common = {
        "apikey": api_key,
        "isOverlayRequired": True,
        "scale": True,
        "detectOrientation": True,
    }

    # Detect image type from magic bytes (jpg vs png) so we can tell OCR.space
    filetype = "JPG"
    if image_bytes[:4] == b"\x89PNG":
        filetype = "PNG"

    attempts = []

    # ── Round 1: URL-based (preferred — fast, no throttling on payload size) ──
    if image_url:
        for engine in (2, 1):
            attempts.append((
                f"URL/Engine {engine}",
                {**base_common, "url": image_url, "OCREngine": engine, "filetype": filetype},
                12,
            ))

    # ── Round 2: base64 fallback ──────────────────────────────────────────────
    mime = "image/jpeg" if filetype == "JPG" else "image/png"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    b64_payload = {**base_common,
                   "base64Image": f"data:{mime};base64,{b64}",
                   "filetype": filetype}
    for engine in (2, 1):
        attempts.append((
            f"b64/Engine {engine}",
            {**b64_payload, "OCREngine": engine},
            25,
        ))

    last_err = None
    for label, payload, timeout_s in attempts:
        try:
            regions = _ocrspace_call(payload, label, timeout_s)
            if regions:
                print(f"[OCR] {label} → {len(regions)} regions")
                return regions
            print(f"[OCR] {label} returned 0 regions — trying next")
            last_err = None
        except requests.exceptions.Timeout as e:
            last_err = e
            print(f"[OCR] {label} timed out — retrying…")
        except Exception as e:
            last_err = e
            print(f"[OCR] {label} error: {e} — retrying…")

    if last_err:
        raise last_err
    return []


def detect_text_google_vision(image_bytes):
    """
    Fallback OCR using Google Cloud Vision REST API.
    Groups word-level annotations into lines by proximity so the output
    format matches the OCR.space line-based structure.
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set — cannot use Vision fallback")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "requests": [{
            "image": {"content": b64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }]
    }
    print("[OCR-GV] Calling Google Vision API…")
    t0 = time.time()
    resp = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
        json=payload,
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    elapsed = time.time() - t0
    print(f"[OCR-GV] Response in {elapsed:.1f}s")

    annotations = data.get("responses", [{}])[0].get("textAnnotations", [])
    if not annotations:
        print("[OCR-GV] No text found")
        return []

    # annotations[0] is the full-page text block; skip it and use word-level ones
    words = annotations[1:]

    # Group words into lines: words whose vertical midpoints are within
    # LINE_GAP pixels of each other are considered the same line.
    LINE_GAP = 12
    lines = []   # list of dicts: {words: [...], min_top, max_bot, min_left, max_right}

    def _bbox_to_rect(vertices):
        xs = [v.get("x", 0) for v in vertices]
        ys = [v.get("y", 0) for v in vertices]
        return min(xs), min(ys), max(xs), max(ys)

    for ann in words:
        verts = ann.get("boundingPoly", {}).get("vertices", [])
        if len(verts) < 4:
            continue
        x1, y1, x2, y2 = _bbox_to_rect(verts)
        mid_y = (y1 + y2) / 2
        placed = False
        for line in lines:
            if abs(mid_y - line["mid_y"]) <= LINE_GAP:
                line["words"].append(ann["description"])
                line["min_left"]  = min(line["min_left"],  x1)
                line["min_top"]   = min(line["min_top"],   y1)
                line["max_right"] = max(line["max_right"], x2)
                line["max_bot"]   = max(line["max_bot"],   y2)
                # Update running midpoint
                line["mid_y"] = (line["min_top"] + line["max_bot"]) / 2
                placed = True
                break
        if not placed:
            lines.append({
                "words": [ann["description"]],
                "min_left": x1, "min_top": y1,
                "max_right": x2, "max_bot": y2,
                "mid_y": mid_y,
            })

    # Sort lines top-to-bottom then build regions
    lines.sort(key=lambda l: l["min_top"])
    text_regions = []
    for i, line in enumerate(lines, start=1):
        text = " ".join(line["words"])
        if not text.strip():
            continue
        text_regions.append({
            "id": i,
            "text": text,
            "x": int(line["min_left"]),
            "y": int(line["min_top"]),
            "width": int(line["max_right"] - line["min_left"]),
            "height": int(line["max_bot"]  - line["min_top"]),
            "type": "straight",
            "confidence": 95.0,
        })

    print(f"[OCR-GV] {len(text_regions)} regions detected")
    return text_regions


def detect_text_with_fallback(image_bytes, image_url=None):
    """
    Primary: OCR.space (URL-based preferred → base64 fallback).
    Fallback: Google Cloud Vision API.
    Returns (regions, source) where source is 'ocrspace' or 'google_vision'.
    """
    try:
        regions = detect_text_ocrspace(image_bytes, image_url=image_url)
        return regions, "ocrspace"
    except Exception as ocr_err:
        print(f"[OCR] OCR.space failed ({ocr_err}) — switching to Google Vision")
        try:
            regions = detect_text_google_vision(image_bytes)
            return regions, "google_vision"
        except Exception as gv_err:
            print(f"[OCR-GV] Google Vision also failed: {gv_err}")
            raise RuntimeError(
                f"All OCR services failed. OCR.space: {ocr_err}. "
                f"Google Vision: {gv_err}"
            )


def find_font(family, bold=False, italic=False):
    for d in EXTRA_FONT_DIRS:
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".ttf", ".otf")):
                    fl = f.lower()
                    fam_lower = family.lower().replace(" ", "")
                    if fam_lower in fl.replace("-", "").replace("_", "").replace(" ", ""):
                        if bold and italic and ("boldital" in fl or "boldoblique" in fl):
                            return os.path.join(root, f)
                        if bold and not italic and "bold" in fl and "ital" not in fl and "oblique" not in fl:
                            return os.path.join(root, f)
                        if italic and not bold and ("italic" in fl or "oblique" in fl) and "bold" not in fl:
                            return os.path.join(root, f)
                        if not bold and not italic and ("regular" in fl or fl.count("-") == 0):
                            return os.path.join(root, f)
    return FONT_PATHS["bold"] if bold else FONT_PATHS["regular"]


def hex_to_rgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def replace_text_in_image(image_bytes, x, y, width, height, new_text, font_size,
                           text_color, bold, italic, letter_spacing, alignment,
                           shadow_enabled, shadow_color, shadow_ox, shadow_oy,
                           outline_enabled, outline_color, outline_width, opacity,
                           font_family="", underline=False, strikethrough=False,
                           all_caps=False, line_height=1.2, background_highlight=False,
                           bg_color="#FFFFFF", bg_padding=4):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    removing = not new_text or not new_text.strip()

    # Small padding — smart mask in _cv_inpaint_region handles anti-aliased edges automatically
    padding = max(4, height // 10)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img.width, x + width + padding)
    y2 = min(img.height, y + height + padding)

    # Gradient-aware background reconstruction — raises if reconstruction fails (no silent fallback)
    if removing:
        img = reconstruct_background_clean(img, x1, y1, x2, y2)
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()
    else:
        img = reconstruct_background(img, x1, y1, x2, y2)

    draw = ImageDraw.Draw(img)

    # Font resolution — 3-tier priority:
    # 1. Pre-downloaded fonts (get_best_font — instant, covers all 7 fonts + aliases)
    # 2. Google Fonts API dynamic download (for any other font name the user requests)
    # 3. Lato fallback (always available, never DejaVu)
    KNOWN_FONTS = {"lato","ubuntu","arvo","tinos","ptsans","pt","anton","bebas","bebasneue",
                   "montserrat","roboto","opensans","oswald","raleway","inter","playfair","merriweather"}
    _fname_key = (font_family or "lato").lower().replace(" ", "").replace("-", "")
    font = None

    if _fname_key in KNOWN_FONTS:
        # Fast path — always works, no network call needed
        font = get_best_font(_fname_key, font_size, bold=bold)
        print(f"[FONT] Using pre-downloaded font: {_fname_key}")
    else:
        # Unknown font — try Google Fonts API download at runtime
        dl_path = download_font_dynamic(font_family or _fname_key, bold=bold)
        if dl_path:
            try:
                font = ImageFont.truetype(dl_path, font_size)
                print(f"[FONT] Using Google Fonts download: {dl_path}")
            except Exception as e:
                print(f"[FONT] Truetype load error for {dl_path}: {e}")
        if font is None:
            # Final fallback — Lato (never DejaVu, never load_default)
            font = get_best_font("lato", font_size, bold=bold)
            print(f"[FONT] Fallback to Lato (could not load '{font_family}')")

    color_tuple = hex_to_rgb(text_color)
    if opacity < 100:
        alpha = int(255 * opacity / 100)
        color_tuple = color_tuple + (alpha,)

    display_text = new_text.upper() if all_caps else new_text

    def calc_text_x(txt):
        bbox_text = draw.textbbox((0, 0), txt, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        if alignment == "center":
            return x + (width - text_w) // 2
        elif alignment == "right":
            return x + width - text_w
        return x

    text_x = calc_text_x(display_text)
    text_y = y

    if background_highlight:
        bg_rgb = hex_to_rgb(bg_color)
        text_bbox = draw.textbbox((0, 0), display_text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        draw.rectangle(
            [text_x - bg_padding, text_y - bg_padding,
             text_x + tw + bg_padding, text_y + th + bg_padding],
            fill=bg_rgb
        )

    if shadow_enabled:
        sr, sg, sb = hex_to_rgb(shadow_color)
        draw.text((text_x + shadow_ox, text_y + shadow_oy), display_text, font=font, fill=(sr, sg, sb))

    if outline_enabled:
        or_, og, ob = hex_to_rgb(outline_color)
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((text_x + dx, text_y + dy), display_text, font=font, fill=(or_, og, ob))

    draw.text((text_x, text_y), display_text, font=font, fill=color_tuple)

    if underline or strikethrough:
        tb = draw.textbbox((0, 0), display_text, font=font)
        tw = tb[2] - tb[0]
        th = tb[3] - tb[1]
        if underline:
            draw.line([(text_x, text_y + th + 2), (text_x + tw, text_y + th + 2)],
                      fill=color_tuple[:3], width=max(1, font_size // 16))
        if strikethrough:
            mid_y = text_y + th // 2
            draw.line([(text_x, mid_y), (text_x + tw, mid_y)],
                      fill=color_tuple[:3], width=max(1, font_size // 16))

    output = io.BytesIO()
    img.save(output, format="PNG")
    output.seek(0)
    return output.getvalue()


# ───────────────────────────── AUTH HELPERS ─────────────────────────────

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


def no_guest(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("guest"):
            return redirect(url_for("login_page") + "?msg=guest")
        if not session.get("user_id"):
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


# ───────────────────────────── PAGE ROUTES ─────────────────────────────

@app.route("/")
def home():
    if session.get("user_id") and not session.get("guest"):
        return redirect(url_for("dashboard"))
    return render_template("landing.html",
                           user=session.get("user_name"),
                           logged_in=bool(session.get("user_id") and not session.get("guest")))


@app.route("/editor")
def editor():
    is_guest = session.get("guest", False)
    is_logged = bool(session.get("user_id") and not is_guest)
    return render_template("editor.html",
                           user_name=session.get("user_name", "Guest"),
                           user_email=session.get("user_email", ""),
                           is_guest=is_guest,
                           is_logged=is_logged,
                           plan=session.get("plan", "free"),
                           project_id=request.args.get("project_id", ""))


@app.route("/login", methods=["GET"])
def login_page():
    if session.get("user_id") and not session.get("guest"):
        return redirect(url_for("dashboard"))
    msg = request.args.get("msg", "")
    return render_template("login.html", msg=msg)


def _update_last_login(user_id):
    """Background-thread: best-effort last_login update — never blocks main request."""
    try:
        c = sqlite3.connect(DB_PATH, timeout=2, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("UPDATE users SET last_login=datetime('now') WHERE id=?", (user_id,))
        c.commit()
        c.close()
    except Exception:
        pass


@app.route("/login", methods=["POST"])
def login_post():
    try:
        if request.is_json:
            body = request.get_json() or {}
            email = body.get("email", "").strip().lower()
            password = body.get("password", "").strip()
        else:
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "").strip()

        if not email or not password:
            return jsonify({"success": False, "error": "Email and password are required."})

        print(f"[LOGIN] Attempt for: {email}")
        t0 = time.time()

        # Read-only SELECT — close immediately after fetch
        conn = get_db()
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        conn.close()

        print(f"[LOGIN] User found: {bool(user)}")
        if not user:
            return jsonify({"success": False, "error": "No account found with this email. Please sign up first."})

        if not check_password_hash(user["password_hashed"], password):
            print(f"[LOGIN] Wrong password for: {email}")
            return jsonify({"success": False, "error": "Incorrect password. Please try again."})

        session.clear()
        session.permanent = True
        session["user_id"] = user["id"]
        session["user_name"] = user["full_name"]
        session["user_email"] = user["email"]
        session["plan"] = user["plan_type"]

        # Fire-and-forget background update — never blocks the response
        threading.Thread(target=_update_last_login, args=(user["id"],), daemon=True).start()

        print(f"[LOGIN] Success for: {email}, id={user['id']} in {time.time()-t0:.3f}s")
        return jsonify({"success": True, "redirect": "/dashboard"})
    except Exception as e:
        print(f"[LOGIN] Error: {e}")
        return jsonify({"success": False, "error": "Something went wrong. Please try again."})


@app.route("/signup", methods=["GET"])
def signup_page():
    if session.get("user_id") and not session.get("guest"):
        return redirect(url_for("dashboard"))
    return render_template("signup.html")


@app.route("/signup", methods=["POST"])
def signup_post():
    try:
        # Accept both JSON and form-data submissions
        if request.is_json:
            body = request.get_json() or {}
            full_name = body.get("full_name", "").strip()
            email = body.get("email", "").strip().lower()
            password = body.get("password", "").strip()
            confirm = body.get("confirm_password", password).strip()
            agree = "on"
        else:
            full_name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "").strip()
            confirm = request.form.get("confirm_password", "").strip()
            agree = request.form.get("agree", "off")

        print(f"[SIGNUP] Attempt: name={full_name}, email={email}")

        if not full_name or not email or not password:
            return jsonify({"success": False, "error": "All fields are required."})
        if password != confirm:
            return jsonify({"success": False, "error": "Passwords do not match."})
        if len(password) < 6:
            return jsonify({"success": False, "error": "Password must be at least 6 characters."})
        if agree != "on":
            return jsonify({"success": False, "error": "You must agree to the Terms of Service."})

        t0 = time.time()
        pw_hash = generate_password_hash(password)

        try:
            conn = get_db()
            c = conn.execute(
                "INSERT INTO users (full_name, email, password_hashed) VALUES (?,?,?)",
                (full_name, email, pw_hash)
            )
            user_id = c.lastrowid
            conn.commit()
            conn.close()
        except sqlite3.IntegrityError:
            print(f"[SIGNUP] Email already exists: {email}")
            return jsonify({"success": False, "error": "An account with that email already exists. Please login instead."})

        session.clear()
        session.permanent = True
        session["user_id"] = user_id
        session["user_name"] = full_name
        session["user_email"] = email
        session["plan"] = "free"

        print(f"[SIGNUP] Created user id={user_id} email={email} in {time.time()-t0:.3f}s — session={dict(session)}")
        return jsonify({"success": True, "redirect": "/dashboard"})
    except Exception as e:
        print(f"[SIGNUP] Error: {e}")
        return jsonify({"success": False, "error": "Something went wrong. Please try again."})


@app.route("/guest", methods=["GET", "POST"])
def guest_login():
    session.clear()
    session["guest"] = True
    session["user_name"] = "Guest User"
    session["guest_exports_remaining"] = 3
    session["guest_since"] = datetime.utcnow().isoformat()
    print("[GUEST] Guest session started")
    return redirect(url_for("editor"))


@app.route("/logout")
def logout():
    print(f"[LOGOUT] User {session.get('user_email', 'guest')} logged out")
    session.clear()
    return redirect(url_for("home"))


@app.route("/dashboard")
@no_guest
def dashboard():
    print(f"[DASHBOARD] session={dict(session)}")
    user_id = session["user_id"]
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    projects = conn.execute(
        "SELECT * FROM projects WHERE user_id=? ORDER BY last_edited_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()

    greeting_hour = datetime.now().hour
    if greeting_hour < 12:
        greeting = "Good morning"
    elif greeting_hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    return render_template("dashboard.html",
                           user=dict(user),
                           projects=[dict(p) for p in projects],
                           greeting=greeting)


@app.route("/settings")
@no_guest
def settings_page():
    user_id = session["user_id"]
    conn = get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    settings_row = conn.execute("SELECT settings_json FROM user_settings WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    prefs = json.loads(settings_row["settings_json"]) if settings_row else {}
    return render_template("settings.html", user=dict(user), prefs=prefs)


# ───────────────────────────── API ROUTES ─────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    t_start = time.time()
    image_bytes = None

    if "image" in request.files:
        file = request.files["image"]
        image_bytes = file.read()
        print(f"[DETECT] File upload: {len(image_bytes)} bytes, filename={file.filename}")
    elif request.is_json or request.content_type == "application/json":
        body = request.get_json(force=True, silent=True) or {}
        image_data = body.get("image_data", "")
        if image_data:
            if image_data.startswith("data:"):
                _, encoded = image_data.split(",", 1)
            else:
                encoded = image_data
            image_bytes = base64.b64decode(encoded)
            print(f"[DETECT] JSON base64: {len(image_bytes)} bytes")

    if not image_bytes:
        return jsonify({"error": "No image provided"}), 400

    # Get original dimensions (and resized bytes for OCR)
    ocr_bytes, orig_width, orig_height, ocr_w, ocr_h = prepare_image_for_ocr(image_bytes)
    print(f"[DETECT] Image dimensions: {orig_width}x{orig_height}")

    # ── Step 1: Upload to Cloudinary (fast, ~0.7 s) ──────────────────────────
    t_up = time.time()
    try:
        res = cloudinary.uploader.upload(io.BytesIO(image_bytes), resource_type="image")
        image_url = res["secure_url"]
        public_id = res["public_id"]
        print(f"[CLOUDINARY] Uploaded in {time.time()-t_up:.2f}s: {image_url[:60]}")
    except Exception as ex:
        print(f"[CLOUDINARY] Upload error: {traceback.format_exc()}")
        return jsonify({"error": f"Image upload failed: {ex}"}), 500

    # ── Step 2: OCR — URL-based first (fastest path), base64 fallback ────────
    t_ocr = time.time()
    try:
        # Pass the Cloudinary URL so OCR.space fetches the image itself
        # (URL requests respond in ~0.7 s vs 25-30 s for base64 uploads)
        text_regions, ocr_source = detect_text_with_fallback(ocr_bytes, image_url=image_url)
        print(f"[OCR] Done via {ocr_source} in {time.time()-t_ocr:.2f}s — {len(text_regions)} regions")
    except Exception as ex:
        print(f"[OCR] Error: {traceback.format_exc()}")
        if isinstance(ex, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            msg = "OCR service timed out. Please try again."
        else:
            msg = f"Text detection failed: {ex}"
        return jsonify({"error": msg}), 500

    try:
        # Scale coordinates back to original image dimensions if we resized
        if ocr_w != orig_width or ocr_h != orig_height:
            sx = orig_width / ocr_w
            sy = orig_height / ocr_h
            for r in text_regions:
                r["x"] = int(r["x"] * sx)
                r["y"] = int(r["y"] * sy)
                r["width"] = int(r["width"] * sx)
                r["height"] = int(r["height"] * sy)
            print(f"[DETECT] Scaled coords by ({sx:.3f}, {sy:.3f}) back to original dims")

        # Detect dominant text color for each region
        orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        for r in text_regions:
            r["detected_color"] = detect_text_color(orig_img, r["x"], r["y"], r["width"], r["height"])
        print(f"[DETECT] Text color detection complete")
    except Exception as e:
        print(f"[DETECT] Post-OCR processing error: {traceback.format_exc()}")
        return jsonify({"error": f"Text detection failed: {str(e)}"}), 500

    print(f"[DETECT] Total time: {time.time()-t_start:.2f}s — returning {len(text_regions)} regions")

    # Auto-save project for logged-in users
    project_id = None
    if session.get("user_id") and not session.get("guest"):
        try:
            conn = get_db()
            c = conn.execute(
                "INSERT INTO projects (user_id, original_image_url, edited_image_url, width, height) VALUES (?,?,?,?,?)",
                (session["user_id"], image_url, image_url, orig_width, orig_height)
            )
            project_id = c.lastrowid
            conn.commit()
            conn.close()
            print(f"[DB] Created project id={project_id}")
        except Exception as e:
            print(f"[DB] Failed to save project: {e}")

    return jsonify({
        "image_url": image_url,
        "public_id": public_id,
        "original_width": orig_width,
        "original_height": orig_height,
        "text_regions": text_regions,
        "project_id": project_id,
    })


@app.route("/rescan", methods=["POST"])
def rescan():
    """Accept base64 image data, run OCR on it, and return updated regions."""
    t_start = time.time()
    try:
        body = request.get_json(force=True, silent=True) or {}
        image_data = body.get("image_data", "")
        if not image_data:
            return jsonify({"success": False, "error": "No image data provided"}), 400

        if image_data.startswith("data:"):
            _, encoded = image_data.split(",", 1)
        else:
            encoded = image_data
        image_bytes = base64.b64decode(encoded)
        print(f"[RESCAN] Received {len(image_bytes)} bytes")

        ocr_bytes, orig_width, orig_height, ocr_w, ocr_h = prepare_image_for_ocr(image_bytes)
        print(f"[RESCAN] Dims: {orig_width}x{orig_height}, OCR dims: {ocr_w}x{ocr_h}")

        # Upload to Cloudinary so we can use URL-based OCR (fast path)
        rescan_image_url = None
        try:
            t_up = time.time()
            res_up = cloudinary.uploader.upload(io.BytesIO(image_bytes), resource_type="image")
            rescan_image_url = res_up["secure_url"]
            print(f"[RESCAN] Cloudinary upload in {time.time()-t_up:.2f}s")
        except Exception as _up_err:
            print(f"[RESCAN] Cloudinary upload skipped: {_up_err} — will use base64 OCR")

        # Run OCR with automatic fallback (URL preferred)
        text_regions, ocr_source = detect_text_with_fallback(ocr_bytes, image_url=rescan_image_url)
        print(f"[RESCAN] OCR via {ocr_source} returned {len(text_regions)} regions")

        # Scale back if image was resized for OCR
        if ocr_w != orig_width or ocr_h != orig_height:
            sx = orig_width / ocr_w
            sy = orig_height / ocr_h
            for r in text_regions:
                r["x"] = int(r["x"] * sx)
                r["y"] = int(r["y"] * sy)
                r["width"] = int(r["width"] * sx)
                r["height"] = int(r["height"] * sy)

        # Color detection
        orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        for r in text_regions:
            try:
                r["detected_color"] = detect_text_color(orig_img, r["x"], r["y"], r["width"], r["height"])
            except Exception:
                r["detected_color"] = "#ffffff"

        # Upload to Cloudinary so the image URL becomes permanent
        try:
            upload_result = cloudinary.uploader.upload(io.BytesIO(image_bytes), resource_type="image")
            new_image_url = upload_result["secure_url"]
            print(f"[RESCAN] Uploaded to Cloudinary: {new_image_url}")
        except Exception as e:
            print(f"[RESCAN] Cloudinary upload failed: {e}")
            new_image_url = None  # frontend will keep the data URL it already has

        print(f"[RESCAN] Done in {time.time()-t_start:.2f}s — {len(text_regions)} regions")
        return jsonify({
            "success": True,
            "image_url": new_image_url,
            "original_width": orig_width,
            "original_height": orig_height,
            "text_regions": text_regions,
        })
    except Exception as e:
        print(f"[RESCAN] Error: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/test-replace")
def test_replace():
    return jsonify({"status": "replace route is working", "success": True})


@app.route("/replace", methods=["POST"])
def replace():
    try:
        # Guest export limit check
        if session.get("guest"):
            remaining = session.get("guest_exports_remaining", 0)
            if remaining <= 0:
                return jsonify({"error": "GUEST_LIMIT", "message": "Guest export limit reached"}), 403
            session["guest_exports_remaining"] = remaining - 1

        body = request.get_json(force=True, silent=True)
        if not body:
            return jsonify({"error": "No data provided"}), 400

        image_url = body.get("image_url")
        if not image_url:
            return jsonify({"error": "No image_url provided"}), 400

        x = int(body.get("x", 0))
        y = int(body.get("y", 0))
        width = int(body.get("width", 100))
        height = int(body.get("height", 30))
        # Original position before any drag — used to erase old text location
        orig_x = int(body.get("original_x", x))
        orig_y = int(body.get("original_y", y))
        orig_w = int(body.get("original_width",  width))
        orig_h = int(body.get("original_height", height))
        new_text = body.get("new_text", "")
        font_size = int(body.get("font_size", 24))
        text_color = body.get("text_color", "#000000")
        bold = bool(body.get("bold", False))
        italic = bool(body.get("italic", False))
        letter_spacing = int(body.get("letter_spacing", 0))
        alignment = body.get("alignment", "left")
        shadow_enabled = bool(body.get("shadow_enabled", False))
        shadow_color = body.get("shadow_color", "#000000")
        shadow_ox = int(body.get("shadow_offset_x", 2))
        shadow_oy = int(body.get("shadow_offset_y", 2))
        outline_enabled = bool(body.get("outline_enabled", False))
        outline_color = body.get("outline_color", "#000000")
        outline_width = int(body.get("outline_width", 1))
        opacity = int(body.get("opacity", 100))
        # Accept both "font_name" (from /apply-all stack) and "font_family" (legacy)
        font_family = body.get("font_name") or body.get("font_family", "")
        underline = bool(body.get("underline", False))
        strikethrough = bool(body.get("strikethrough", False))
        all_caps = bool(body.get("all_caps", False))
        line_height = float(body.get("line_height", 1.2))
        background_highlight = bool(body.get("background_highlight", False))
        bg_color = body.get("bg_color", "#FFFFFF")
        bg_padding = int(body.get("bg_padding", 4))
        detected_color = body.get("detected_color", None)
        project_id = body.get("project_id")

        moved = abs(x - orig_x) > 5 or abs(y - orig_y) > 5
        print(f"[REPLACE] region=({x},{y},{width},{height}) orig=({orig_x},{orig_y}) moved={moved} text='{new_text}' font={font_family}:{font_size}px color={text_color}")

        t_start = time.time()

        # Support both Cloudinary URLs and base64 data URLs from prior edits
        try:
            if image_url.startswith("data:"):
                header, b64data = image_url.split(",", 1)
                image_bytes = base64.b64decode(b64data)
                print(f"[REPLACE] Decoded data URL: {len(image_bytes)} bytes")
            else:
                t_dl = time.time()
                img_resp = requests.get(image_url, timeout=30)
                img_resp.raise_for_status()
                image_bytes = img_resp.content
                print(f"[REPLACE] Downloaded image in {time.time()-t_dl:.2f}s: {len(image_bytes)} bytes")
        except Exception as e:
            print(f"[REPLACE] Image load error: {traceback.format_exc()}")
            return jsonify({"error": f"Failed to load image: {str(e)}"}), 400

        # Fast replace with gradient-aware background and auto color/size
        try:
            t_proc = time.time()
            result_bytes = replace_text_fast(
                image_bytes, x, y, width, height, new_text, font_size,
                text_color, bold, italic=italic, font_family=font_family,
                detected_color=detected_color, alignment=alignment,
                letter_spacing=letter_spacing,
                orig_x=orig_x, orig_y=orig_y, orig_w=orig_w, orig_h=orig_h
            )
            print(f"[REPLACE] Fast replace done in {time.time()-t_proc:.3f}s: {len(result_bytes)} bytes")
        except Exception as e:
            print(f"[REPLACE] Processing error: {traceback.format_exc()}")
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

        # Return base64 data URL directly — no Cloudinary upload needed for editing
        result_b64 = base64.b64encode(result_bytes).decode("utf-8")
        data_url = "data:image/png;base64," + result_b64
        print(f"[REPLACE] Total time: {time.time()-t_start:.2f}s — returning base64 ({len(result_b64)} chars)")

        return jsonify({
            "edited_image_url": data_url,
            "guest_exports_remaining": session.get("guest_exports_remaining"),
            "saved": False,
        })

    except Exception as e:
        print(f"[REPLACE] Unhandled error: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}", "details": traceback.format_exc()}), 500


@app.route("/remove", methods=["POST"])
def remove_text():
    try:
        if session.get("guest"):
            remaining = session.get("guest_exports_remaining", 0)
            if remaining <= 0:
                return jsonify({"error": "GUEST_LIMIT", "message": "Guest export limit reached"}), 403
            session["guest_exports_remaining"] = remaining - 1

        body = request.get_json(force=True, silent=True)
        if not body:
            return jsonify({"error": "No data provided"}), 400

        image_url = body.get("image_url")
        if not image_url:
            return jsonify({"error": "No image_url provided"}), 400

        x = int(body.get("x", 0))
        y = int(body.get("y", 0))
        width = int(body.get("width", 100))
        height = int(body.get("height", 30))
        project_id = body.get("project_id")

        print(f"[REMOVE] region=({x},{y},{width},{height}) from {image_url[:60]}")

        # Support both Cloudinary URLs and base64 data URLs
        try:
            if image_url.startswith("data:"):
                header, b64data = image_url.split(",", 1)
                image_bytes = base64.b64decode(b64data)
                print(f"[REMOVE] Decoded data URL: {len(image_bytes)} bytes")
            else:
                img_resp = requests.get(image_url, timeout=30)
                img_resp.raise_for_status()
                image_bytes = img_resp.content
                print(f"[REMOVE] Downloaded image: {len(image_bytes)} bytes")
        except Exception as e:
            print(f"[REMOVE] Image load error: {traceback.format_exc()}")
            return jsonify({"error": f"Failed to load image: {str(e)}"}), 400

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        padding = 8
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.width, x + width + padding)
        y2 = min(img.height, y + height + padding)

        arr = np.array(img, dtype=np.uint8)
        result_arr = _reconstruct_bg_coons(arr, x1, y1, x2, y2)
        clean_img = Image.fromarray(result_arr)

        output = io.BytesIO()
        clean_img.save(output, format="PNG")
        output.seek(0)

        # Return as base64 data URL (same pattern as /replace — no Cloudinary needed)
        result_b64 = base64.b64encode(output.getvalue()).decode("utf-8")
        data_url = "data:image/png;base64," + result_b64
        print(f"[REMOVE] Done — returning base64 ({len(result_b64)} chars)")

        return jsonify({
            "edited_image_url": data_url,
            "guest_exports_remaining": session.get("guest_exports_remaining"),
        })

    except Exception as e:
        print(f"[REMOVE] Unhandled error: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}", "details": traceback.format_exc()}), 500


def fill_background(arr, x1, y1, x2, y2):
    """
    Background reconstruction using Coons-patch interpolation.
    Small regions: single pass.  Large regions: multi-pass iterative fill.
    """
    rh = max(0, y2 - y1)
    rw = max(0, x2 - x1)
    if rh > 50 or rw > 150:
        return _reconstruct_bg_multipass(arr, x1, y1, x2, y2)
    return _reconstruct_bg_coons(arr, x1, y1, x2, y2)


@app.route("/apply-all", methods=["POST"])
def apply_all():
    """Re-render ALL edits from the original clean image — permanently fixes duplicate-text stacking."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "No data"}), 400

        original_url = data.get("original_url", "")
        edits = data.get("edits", [])
        print(f"[APPLY-ALL] {len(edits)} edit(s) — url: {original_url[:60]}")

        if not original_url:
            return jsonify({"success": False, "error": "No original_url provided"}), 400

        # Load the original image (always the clean uploaded image, never an edited version)
        if original_url.startswith("data:"):
            _, encoded = original_url.split(",", 1)
            image_bytes = base64.b64decode(encoded)
        else:
            resp = requests.get(original_url, timeout=30)
            resp.raise_for_status()
            image_bytes = resp.content

        orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_w, img_h = orig_img.size
        arr = np.array(orig_img).copy()

        for i, edit in enumerate(edits):
            ox  = int(edit.get("orig_x", edit.get("x", 0)))
            oy  = int(edit.get("orig_y", edit.get("y", 0)))
            ow  = int(edit.get("orig_w", edit.get("w", 100)))
            oh  = int(edit.get("orig_h", edit.get("h", 50)))
            nx  = int(edit.get("x", ox))
            ny  = int(edit.get("y", oy))
            nw  = int(edit.get("w", ow))
            nh  = int(edit.get("h", oh))
            text          = str(edit.get("text", "")).strip()
            fsize_raw     = edit.get("font_size")
            tcolor        = edit.get("text_color", "#ffffff")
            bold          = bool(edit.get("bold", False))
            all_caps      = bool(edit.get("all_caps", False))
            align         = edit.get("alignment", "left")
            fname         = (edit.get("font_name") or edit.get("font_family") or "lato").lower().replace(" ", "").replace("-", "")
            italic        = bool(edit.get("italic", False))
            underline     = bool(edit.get("underline", False))
            strikethrough = bool(edit.get("strikethrough", False))
            outline_on    = bool(edit.get("outline_enabled", False))
            outline_col   = edit.get("outline_color", "#000000")
            outline_w     = int(edit.get("outline_width", 1))
            shadow_on     = bool(edit.get("shadow_enabled", False))
            shadow_col    = edit.get("shadow_color", "#000000")
            shadow_ox     = int(edit.get("shadow_offset_x", edit.get("shadow_ox", 2)))
            shadow_oy     = int(edit.get("shadow_offset_y", edit.get("shadow_oy", 2)))

            # Step 1: erase original OCR bbox — small padding, smart mask handles edges
            pad = max(4, oh // 10)
            arr = fill_background(arr, ox - pad, oy - pad, ox + ow + pad, oy + oh + pad)
            print(f"[APPLY-ALL] edit {i+1}: erased orig bbox ({ox},{oy},{ow},{oh}) pad={pad}")

            if not text:
                continue  # remove-only edit

            # Step 2: resolve font — 3-tier: pre-downloaded → Google Fonts API → Lato
            auto_sz = max(8, int(nh * 0.75))
            fsz = max(8, min(200, int(fsize_raw) if fsize_raw else auto_sz))
            KNOWN_FONTS = {"lato","ubuntu","arvo","tinos","ptsans","pt","anton","bebas","bebasneue",
                           "montserrat","roboto","opensans","oswald","raleway","inter","playfair","merriweather"}
            if fname in KNOWN_FONTS:
                font = get_best_font(fname, fsz, bold=bold)
            else:
                dl_path = download_font_dynamic(fname, bold=bold)
                if dl_path:
                    try:
                        font = ImageFont.truetype(dl_path, fsz)
                    except Exception:
                        font = get_best_font("lato", fsz, bold=bold)
                else:
                    font = get_best_font("lato", fsz, bold=bold)

            # Step 3: draw new text at the target position
            img_tmp = Image.fromarray(arr)
            drw = ImageDraw.Draw(img_tmp)
            display_text = text.upper() if all_caps else text
            rgb = hex_to_rgb(tcolor)

            try:
                bb = drw.textbbox((0, 0), display_text, font=font)
                tw = bb[2] - bb[0]
                th = bb[3] - bb[1]
            except Exception:
                tw = len(display_text) * fsz // 2
                th = fsz

            if align == "center":
                tx = nx + (nw - tw) // 2
            elif align == "right":
                tx = nx + nw - tw
            else:
                tx = nx
            tx = max(0, min(img_w - 1, tx))
            ty = max(0, min(img_h - 1, ny))

            if shadow_on:
                s_rgb = hex_to_rgb(shadow_col)
                drw.text((tx + shadow_ox, ty + shadow_oy), display_text, font=font, fill=s_rgb)
            if outline_on:
                o_rgb = hex_to_rgb(outline_col)
                ow2 = max(1, outline_w)
                for ddx in range(-ow2, ow2 + 1):
                    for ddy in range(-ow2, ow2 + 1):
                        if ddx != 0 or ddy != 0:
                            drw.text((tx + ddx, ty + ddy), display_text, font=font, fill=o_rgb)
            drw.text((tx, ty), display_text, font=font, fill=rgb)

            # Underline / strikethrough decorations
            if underline:
                drw.line([(tx, ty + th + 2), (tx + tw, ty + th + 2)],
                         fill=rgb, width=max(1, fsz // 16))
            if strikethrough:
                mid_y = ty + th // 2
                drw.line([(tx, mid_y), (tx + tw, mid_y)],
                         fill=rgb, width=max(1, fsz // 16))

            arr = np.array(img_tmp)
            print(f"[APPLY-ALL] edit {i+1}: drew '{display_text}' font={fname} fsz={fsz} at ({tx},{ty}) color={rgb}")

        out = io.BytesIO()
        Image.fromarray(arr).save(out, format="PNG", optimize=True)
        out.seek(0)
        b64 = base64.b64encode(out.getvalue()).decode("utf-8")
        print(f"[APPLY-ALL] complete — {len(edits)} edit(s), result {len(b64)} chars")
        return jsonify({
            "success": True,
            "edited_image_url": "data:image/png;base64," + b64,
            "guest_exports_remaining": session.get("guest_exports_remaining"),
        })

    except Exception as e:
        print(f"[APPLY-ALL] ERROR: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e), "details": traceback.format_exc()}), 500


@app.route("/export", methods=["POST"])
def export_image():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400
        image_url = data.get("image_url", "")
        fmt = data.get("format", "PNG").upper()
        quality = int(data.get("quality", 95))
        resolution = data.get("resolution", "original")
        filename = data.get("filename", "textora-edited")

        if image_url.startswith("data:"):
            header, b64data = image_url.split(",", 1)
            img = Image.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")
        else:
            resp = requests.get(image_url, timeout=15)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        if resolution == "hd":
            ratio = 1920 / img.width
            img = img.resize((1920, int(img.height * ratio)), Image.LANCZOS)
        elif resolution == "2x":
            img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        elif resolution == "custom":
            cw = int(data.get("custom_width", img.width))
            ch = int(data.get("custom_height", img.height))
            img = img.resize((cw, ch), Image.LANCZOS)

        output = io.BytesIO()
        if fmt in ("JPG", "JPEG"):
            img.save(output, format="JPEG", quality=quality, optimize=True)
            mime = "image/jpeg"; ext = "jpg"
        elif fmt == "WEBP":
            img.save(output, format="WEBP", quality=quality)
            mime = "image/webp"; ext = "webp"
        else:
            img.save(output, format="PNG", optimize=True)
            mime = "image/png"; ext = "png"

        output.seek(0)
        b64 = base64.b64encode(output.getvalue()).decode("utf-8")
        return jsonify({
            "success": True,
            "data_url": f"data:{mime};base64,{b64}",
            "filename": f"{filename}.{ext}",
            "mime": mime
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/projects", methods=["GET"])
@no_guest
def api_projects():
    conn = get_db()
    projects = conn.execute(
        "SELECT * FROM projects WHERE user_id=? ORDER BY last_edited_at DESC",
        (session["user_id"],)
    ).fetchall()
    conn.close()
    return jsonify({"projects": [dict(p) for p in projects]})


@app.route("/api/projects/<int:pid>", methods=["PATCH"])
@no_guest
def api_rename_project(pid):
    body = request.get_json()
    name = body.get("project_name", "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400
    conn = get_db()
    conn.execute("UPDATE projects SET project_name=? WHERE id=? AND user_id=?",
                 (name, pid, session["user_id"]))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/projects/<int:pid>", methods=["DELETE"])
@no_guest
def api_delete_project(pid):
    conn = get_db()
    conn.execute("DELETE FROM projects WHERE id=? AND user_id=?", (pid, session["user_id"]))
    conn.commit()
    conn.close()
    print(f"[DB] Deleted project id={pid}")
    return jsonify({"ok": True})


@app.route("/api/profile", methods=["POST"])
@no_guest
def api_save_profile():
    body = request.get_json()
    full_name = body.get("full_name", "").strip()
    bio = body.get("bio", "").strip()[:160]
    website = body.get("website", "").strip()
    if not full_name:
        return jsonify({"error": "Name required"}), 400
    conn = get_db()
    conn.execute("UPDATE users SET full_name=?, bio=?, website=? WHERE id=?",
                 (full_name, bio, website, session["user_id"]))
    conn.commit()
    conn.close()
    session["user_name"] = full_name
    print(f"[PROFILE] Updated profile for user {session['user_id']}")
    return jsonify({"ok": True, "full_name": full_name})


@app.route("/api/password", methods=["POST"])
@no_guest
def api_change_password():
    body = request.get_json()
    current = body.get("current_password", "")
    new_pw = body.get("new_password", "")
    if len(new_pw) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    conn = get_db()
    user = conn.execute("SELECT password_hashed FROM users WHERE id=?", (session["user_id"],)).fetchone()
    if not check_password_hash(user["password_hashed"], current):
        conn.close()
        return jsonify({"error": "Current password is incorrect"}), 400
    conn.execute("UPDATE users SET password_hashed=? WHERE id=?",
                 (generate_password_hash(new_pw), session["user_id"]))
    conn.commit()
    conn.close()
    print(f"[PASSWORD] Changed for user {session['user_id']}")
    return jsonify({"ok": True})


@app.route("/api/settings", methods=["POST"])
@no_guest
def api_save_settings():
    body = request.get_json()
    conn = get_db()
    existing = conn.execute("SELECT user_id FROM user_settings WHERE user_id=?",
                            (session["user_id"],)).fetchone()
    if existing:
        conn.execute("UPDATE user_settings SET settings_json=? WHERE user_id=?",
                     (json.dumps(body), session["user_id"]))
    else:
        conn.execute("INSERT INTO user_settings (user_id, settings_json) VALUES (?,?)",
                     (session["user_id"], json.dumps(body)))
    conn.commit()
    conn.close()
    print(f"[SETTINGS] Saved settings for user {session['user_id']}")
    return jsonify({"ok": True})


@app.route("/api/delete-all-projects", methods=["POST"])
@no_guest
def api_delete_all_projects():
    conn = get_db()
    conn.execute("DELETE FROM projects WHERE user_id=?", (session["user_id"],))
    conn.commit()
    conn.close()
    print(f"[DB] Deleted all projects for user {session['user_id']}")
    return jsonify({"ok": True})


@app.route("/api/delete-account", methods=["POST"])
@no_guest
def api_delete_account():
    body = request.get_json()
    email_confirm = body.get("email", "").strip().lower()
    if email_confirm != session.get("user_email", "").lower():
        return jsonify({"error": "Email does not match"}), 400
    uid = session["user_id"]
    conn = get_db()
    conn.execute("DELETE FROM projects WHERE user_id=?", (uid,))
    conn.execute("DELETE FROM user_settings WHERE user_id=?", (uid,))
    conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    session.clear()
    print(f"[DB] Deleted account id={uid}")
    return jsonify({"ok": True})


@app.route("/api/save-project", methods=["POST"])
@no_guest
def api_save_project():
    body = request.get_json()
    project_id = body.get("project_id")
    edited_url = body.get("edited_image_url", "")
    project_name = body.get("project_name", "")

    if project_id:
        conn = get_db()
        if project_name:
            conn.execute(
                "UPDATE projects SET edited_image_url=?, project_name=?, last_edited_at=datetime('now') WHERE id=? AND user_id=?",
                (edited_url, project_name, project_id, session["user_id"])
            )
        else:
            conn.execute(
                "UPDATE projects SET edited_image_url=?, last_edited_at=datetime('now') WHERE id=? AND user_id=?",
                (edited_url, project_id, session["user_id"])
            )
        conn.commit()
        conn.close()
        return jsonify({"ok": True, "project_id": project_id})
    return jsonify({"error": "No project_id"}), 400


def _kill_port(port):
    """Kill any process listening on the given TCP port using /proc/net/tcp."""
    import signal as _sig
    try:
        hex_port = format(port, '04X')
        with open('/proc/net/tcp') as f:
            lines = f.readlines()[1:]
        inodes = set()
        for line in lines:
            fields = line.strip().split()
            if len(fields) < 10:
                continue
            local = fields[1]
            if local.split(':')[1].upper() == hex_port:
                inodes.add(fields[9])
        for inode in inodes:
            for pid_str in os.listdir('/proc'):
                if not pid_str.isdigit():
                    continue
                try:
                    for fd in os.listdir(f'/proc/{pid_str}/fd'):
                        try:
                            link = os.readlink(f'/proc/{pid_str}/fd/{fd}')
                            if f'socket:[{inode}]' in link:
                                pid = int(pid_str)
                                if pid != os.getpid():
                                    os.kill(pid, _sig.SIGTERM)
                                    print(f"[PORT-KILL] Sent SIGTERM to pid {pid} on port {port}")
                        except OSError:
                            pass
                except OSError:
                    pass
    except Exception as ex:
        print(f"[PORT-KILL] Warning: {ex}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Kill any stale process on the port before binding (prevents EADDRINUSE on restart)
    _kill_port(port)
    time.sleep(0.5)
    print(f"[SERVER] Starting Textora on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
