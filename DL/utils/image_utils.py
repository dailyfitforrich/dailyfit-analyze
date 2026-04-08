# utils/image_utils.py
# ============================================================
# DailyFit — 圖像工具函式
#
# 字型策略（確保跨平台正確顯示中文 + 角度符號°）：
#   1. 優先使用專案內 assets/font.ttc（WQY 文泉驛，已打包，跨平台）
#   2. 若不存在則嘗試系統字型（Linux / macOS / Windows 常見路徑）
#   3. 最終 fallback：PIL 內建 bitmap 字型（ASCII 可讀，中文顯示為方塊）
#
# OpenCV putText 完全不支援 Unicode，所有文字均使用 PIL ImageDraw 渲染。
# ============================================================

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
from typing import Optional, Tuple

from utils.constants import ANNOTATION_WIDTH, ANNOTATION_HEIGHT

# ── 字型搜尋（優先順序：打包 → 系統 → fallback）─────────

def _find_font() -> Optional[str]:
    """
    跨平台字型搜尋。
    回傳第一個存在的字型路徑，或 None（使用 PIL 內建字型）。
    """
    candidates = [
        # 1. 專案打包字型（最高優先，跨平台保證）
        os.path.join(os.path.dirname(__file__), "..", "assets", "font.ttc"),

        # 2. Linux — Noto / WQY
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",

        # 3. macOS 中文字型
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",

        # 4. Windows 中文字型
        "C:/Windows/Fonts/msjh.ttc",        # 微軟正黑體（繁體）
        "C:/Windows/Fonts/msjhbd.ttc",
        "C:/Windows/Fonts/kaiu.ttf",         # 標楷體（繁體）
        "C:/Windows/Fonts/mingliu.ttc",      # 細明體（繁體）
        "C:/Windows/Fonts/msyh.ttc",         # 微軟雅黑（簡體）
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/simhei.ttf",

        # 5. Linux 其他路徑
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # 無中文，但不顯示方塊
    ]
    for path in candidates:
        norm = os.path.normpath(path)
        if os.path.exists(norm):
            return norm
    return None


_FONT_PATH: Optional[str] = _find_font()

# 預載字型快取
_FONT_CACHE: dict = {}


def get_font(size: int) -> ImageFont.FreeTypeFont:
    """取得指定大小的 PIL 字型物件（快取）"""
    if size not in _FONT_CACHE:
        if _FONT_PATH:
            try:
                _FONT_CACHE[size] = ImageFont.truetype(_FONT_PATH, size)
            except Exception:
                _FONT_CACHE[size] = ImageFont.load_default()
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


# ── BGR ↔ PIL 互轉 ─────────────────────────────────────────

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """PIL RGB → OpenCV BGR"""
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """OpenCV BGR → PIL RGB"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


# ── PIL 文字渲染（核心，支援中文 + °）────────────────────

def draw_text_pil(
    img_bgr: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_size: int = 13,
    text_color: Tuple[int, int, int] = (255, 255, 255),   # RGB
    bg_color: Optional[Tuple[int, int, int]] = (25, 25, 25),
    padding: int = 4,
    opacity: float = 0.88,
) -> np.ndarray:
    """
    使用 PIL 在 OpenCV BGR 圖像上繪製 Unicode 文字。
    回傳修改後的 BGR ndarray（不就地修改）。
    """
    if not text:
        return img_bgr

    pil = cv2_to_pil(img_bgr)
    draw = ImageDraw.Draw(pil, "RGBA")
    font = get_font(font_size)
    x, y = pos

    # 量測文字尺寸（相容 PIL 新舊版本）
    try:
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        tw, th = font.getsize(text)

    # 半透明背景矩形
    if bg_color is not None:
        alpha = int(255 * opacity)
        draw.rectangle(
            [x - padding, y - padding, x + tw + padding * 2, y + th + padding],
            fill=(*bg_color, alpha),
        )

    draw.text((x, y), text, font=font, fill=(*text_color, 255))
    return pil_to_cv2(pil)


def draw_label(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_size: int = 13,
    color: Tuple[int, int, int] = (255, 255, 255),   # BGR
    bg_color: Tuple[int, int, int] = (30, 30, 30),   # BGR
    **kwargs,
) -> None:
    """
    就地在 BGR 圖像上繪製標籤（中文 + 角度符號均可正確顯示）。
    color / bg_color 以 BGR 傳入，內部自動轉換為 RGB。
    """
    rgb_color = (color[2],    color[1],    color[0])
    rgb_bg    = (bg_color[2], bg_color[1], bg_color[0])
    result = draw_text_pil(img, text, pos,
                           font_size=font_size,
                           text_color=rgb_color,
                           bg_color=rgb_bg)
    img[:] = result[:]


# ── Base64 輸出（HTML 報告內嵌圖片用）────────────────────

def image_to_base64(image: np.ndarray, fmt: str = "JPEG") -> str:
    """OpenCV BGR ndarray → base64 data URI"""
    pil_img = cv2_to_pil(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format=fmt, quality=90)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def pil_to_base64(pil_img: Image.Image, fmt: str = "JPEG") -> str:
    """PIL Image → base64 data URI"""
    buffer = io.BytesIO()
    pil_img.convert("RGB").save(buffer, format=fmt, quality=90)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buffer.getvalue()).decode()}"


# ── 安全地標存取 ──────────────────────────────────────────

def safe_lm(landmarks, idx: int, visibility_threshold: float = 0.4):
    """安全取得 MediaPipe 地標，低信心時回傳 None"""
    if landmarks is None or idx >= len(landmarks.landmark):
        return None
    lm = landmarks.landmark[idx]
    if lm.visibility < visibility_threshold:
        return None
    return lm
