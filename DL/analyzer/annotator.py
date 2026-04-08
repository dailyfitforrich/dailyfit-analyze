# analyzer/annotator.py
# ============================================================
# DailyFit — 圖像標注模組（v3.2 修正）
#
# 修正重點：
#   - 所有文字改用 PIL 渲染（支援中文標籤 + 角度符號°）
#   - 角度數值直接顯示在標注線旁
#   - 不再出現 ???? 亂碼
# ============================================================

import cv2
import numpy as np
import math
from typing import Optional, Tuple
from PIL import Image, ImageDraw

from analyzer.pose_detector import DetectionResult, LM
from utils.constants import *
from utils.image_utils import (
    draw_label, draw_text_pil,
    cv2_to_pil, pil_to_cv2, get_font,
)


# ── 顏色定義（BGR，供 OpenCV 幾何圖形使用）───────────────
C_LANDMARK   = (0,   255, 136)   # 螢光綠
C_SHOULDER   = (0,   215, 255)   # 金黃
C_HIP        = (80,  107, 255)   # 珊瑚紅
C_MEASURE    = (255, 210,  70)   # 青藍
C_SPINE      = (200, 100, 255)   # 紫
C_WHITE      = (255, 255, 255)
C_ANGLE      = (0,   165, 255)   # 橙（角度）
C_GRAY       = (180, 180, 180)

# 對應 PIL RGB 版本（draw_text_pil 需要 RGB）
PC_SHOULDER  = (255, 215,   0)
PC_HIP       = (255, 107,  80)
PC_MEASURE   = ( 70, 210, 255)
PC_SPINE     = (255, 100, 200)
PC_ANGLE     = (255, 165,   0)
PC_WHITE     = (255, 255, 255)
PC_GREEN     = (100, 255, 160)
PC_CYAN      = (100, 220, 255)


# ── 幾何輔助 ─────────────────────────────────────────────

def _mid(p1, p2):
    if p1 is None or p2 is None:
        return None
    return (int((p1[0]+p2[0])//2), int((p1[1]+p2[1])//2))


def _resize_canvas(img: np.ndarray):
    """等比縮放並填充深灰背景，回傳 (canvas, (x_off, y_off, scale))"""
    h, w = img.shape[:2]
    scale = min(ANNOTATION_WIDTH / w, ANNOTATION_HEIGHT / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas  = np.full((ANNOTATION_HEIGHT, ANNOTATION_WIDTH, 3), 22, dtype=np.uint8)
    y_off   = (ANNOTATION_HEIGHT - nh) // 2
    x_off   = (ANNOTATION_WIDTH  - nw) // 2
    canvas[y_off:y_off+nh, x_off:x_off+nw] = resized
    return canvas, (x_off, y_off, scale)


def _landmark_px(result: DetectionResult, scale: float,
                 x_off: int, y_off: int, idx: int) -> Optional[Tuple[int,int]]:
    """MediaPipe 正規化地標 → 畫布像素座標"""
    pt = result.px(idx)
    if pt is None:
        return None
    return (int(pt[0]*scale) + x_off, int(pt[1]*scale) + y_off)


def _draw_lm(img, pt, color=C_LANDMARK, r=6):
    """繪製地標圓點（帶白邊）"""
    if pt:
        cv2.circle(img, pt, r,   color,   -1, cv2.LINE_AA)
        cv2.circle(img, pt, r+2, C_WHITE,  1, cv2.LINE_AA)


def _draw_line(img, p1, p2, color, thickness=2):
    if p1 and p2:
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def _draw_dashed(img, p1, p2, color, thickness=1, gap=12):
    """繪製虛線"""
    if p1 is None or p2 is None:
        return
    d = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    if d < 1:
        return
    n = max(1, int(d / gap))
    for i in range(n):
        if i % 2 == 0:
            x1 = int(p1[0]+(p2[0]-p1[0])*i/n)
            y1 = int(p1[1]+(p2[1]-p1[1])*i/n)
            x2 = int(p1[0]+(p2[0]-p1[0])*(i+1)/n)
            y2 = int(p1[1]+(p2[1]-p1[1])*(i+1)/n)
            cv2.line(img, (x1,y1), (x2,y2), color, thickness, cv2.LINE_AA)


def _angle_arc(img, vertex, p1, p3, radius=32):
    """在 vertex 處繪製 p1→vertex→p3 的角度弧線"""
    if vertex is None or p1 is None or p3 is None:
        return
    v1 = np.array([p1[0]-vertex[0], p1[1]-vertex[1]], float)
    v2 = np.array([p3[0]-vertex[0], p3[1]-vertex[1]], float)
    a1 = math.degrees(math.atan2(-v1[1], v1[0]))
    a2 = math.degrees(math.atan2(-v2[1], v2[0]))
    if a1 > a2:
        a1, a2 = a2, a1
    cv2.ellipse(img, vertex, (radius,radius), 0, -a2, -a1, C_ANGLE, 2, cv2.LINE_AA)


# ── PIL 疊字輔助（角度數字 + 中文標籤）──────────────────

def _label(img: np.ndarray, text: str, pos: Tuple[int,int],
           color_rgb=(255,255,255), size=13, bg=(25,25,25)) -> None:
    """PIL 渲染標籤（支援中文 + °，就地更新 img）"""
    draw_label(img, text, pos,
               font_size=size,
               color=(color_rgb[2], color_rgb[1], color_rgb[0]),   # 傳 BGR
               bg_color=(bg[2], bg[1], bg[0]))


def _angle_label(img: np.ndarray, angle: Optional[float],
                 prefix: str, pos: Tuple[int,int],
                 color_rgb=PC_ANGLE, size=13) -> None:
    """在圖像上顯示角度數值，格式如 '頸椎角 17.3°'"""
    if angle is None:
        return
    text = f"{prefix} {angle:.1f}\u00b0"   # \u00b0 = °
    _label(img, text, pos, color_rgb=color_rgb, size=size)


def _watermark(img: np.ndarray, text: str) -> None:
    """底部浮水印橫條（中文標題）"""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h-32), (w, h), (18,18,18), -1)
    _label(img, f"DailyFit  |  {text}", (10, h-26),
           color_rgb=(140,140,140), size=12, bg=(18,18,18))


def _draw_legend(img: np.ndarray,
                 items: list) -> None:
    """
    在右上角繪製乾淨的顏色圖例。
    items = [("肩線", BGR_COLOR), ("髖線", BGR_COLOR), ...]
    用小色點 + 文字，整齊排列，不干擾照片主體。
    """
    h, w = img.shape[:2]
    pad_x, pad_y = 8, 8
    item_h = 20
    box_w  = 90
    box_h  = pad_y * 2 + item_h * len(items)

    # 半透明深色背景
    x0 = w - box_w - 6
    y0 = pad_y
    overlay = img.copy()
    cv2.rectangle(overlay, (x0-4, y0-4),
                  (x0 + box_w, y0 + box_h), (15,15,15), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    for i, (label_text, bgr_color) in enumerate(items):
        cy = y0 + pad_y + i * item_h + item_h // 2
        cx = x0 + 8
        # 色點
        cv2.circle(img, (cx, cy), 5, bgr_color, -1, cv2.LINE_AA)
        # 文字（PIL 渲染，支援中文）
        rgb = (bgr_color[2], bgr_color[1], bgr_color[0])
        _label(img, label_text, (cx + 12, cy - 8),
               color_rgb=rgb, size=12, bg=(15,15,15))


def _info_banner(img: np.ndarray, text: str,
                 bg_rgb=(0,60,160)) -> None:
    """頂部資訊橫條（中文提示）"""
    bg_bgr = (bg_rgb[2], bg_rgb[1], bg_rgb[0])
    cv2.rectangle(img, (0,0), (img.shape[1],30), bg_bgr, -1)
    _label(img, text, (10,7), color_rgb=PC_WHITE, size=13, bg=bg_rgb)


# ═══════════════════════════════════════════════════════════
# 各面向標注函式
# ═══════════════════════════════════════════════════════════

def annotate_front_view(result: DetectionResult, findings: dict) -> np.ndarray:
    """正面站姿標注：肩線、髖線、不對稱標示"""
    img, (x_off, y_off, scale) = _resize_canvas(result.cv2_image.copy())

    def rp(idx): return _landmark_px(result, scale, x_off, y_off, idx)

    l_sh  = rp(LM.LEFT_SHOULDER);  r_sh  = rp(LM.RIGHT_SHOULDER)
    l_hip = rp(LM.LEFT_HIP);       r_hip = rp(LM.RIGHT_HIP)
    l_kn  = rp(LM.LEFT_KNEE);      r_kn  = rp(LM.RIGHT_KNEE)
    l_ank = rp(LM.LEFT_ANKLE);     r_ank = rp(LM.RIGHT_ANKLE)
    nose  = rp(LM.NOSE)
    l_ear = rp(LM.LEFT_EAR);       r_ear = rp(LM.RIGHT_EAR)

    # ── 鉛直中線 ──
    if nose:
        _draw_dashed(img, (nose[0], 0), (nose[0], ANNOTATION_HEIGHT),
                     (120,120,120), 1, 18)

    # ── 肩線 ──
    _draw_line(img, l_sh, r_sh, C_SHOULDER, 3)
    sh_mid = _mid(l_sh, r_sh)

    # ── 髖線 ──
    _draw_line(img, l_hip, r_hip, C_HIP, 3)
    hip_mid = _mid(l_hip, r_hip)

    # ── 軀幹虛線 ──
    _draw_dashed(img, sh_mid, hip_mid, (180,180,180), 1, 8)

    # ── 腿部線段 ──
    for p1, p2 in [(l_hip,l_kn),(r_hip,r_kn),(l_kn,l_ank),(r_kn,r_ank)]:
        _draw_line(img, p1, p2, C_GRAY, 1)

    # ── 地標 ──
    for pt in [l_sh, r_sh, l_hip, r_hip, l_kn, r_kn, l_ank, r_ank]:
        _draw_lm(img, pt)
    for pt in [l_ear, r_ear, nose]:
        _draw_lm(img, pt, color=(200,200,255), r=4)

    # ── 肩部不對稱標示（含高低差資訊）──
    sh_asym = findings.get("shoulder_asymmetry", {})
    _mark_asymmetry(img, l_sh, r_sh, sh_asym, "肩")

    # ── 骨盆不對稱標示 ──
    hip_asym = findings.get("pelvic_asymmetry", {})
    _mark_asymmetry(img, l_hip, r_hip, hip_asym, "髖")

    _watermark(img, "正面姿勢分析")
    _draw_legend(img, [("肩線", C_SHOULDER), ("髖線", C_HIP)])
    return img


def annotate_back_view(result: DetectionResult, findings: dict) -> np.ndarray:
    """背面站姿標注：脊柱參考線、肩線、髖線"""
    img, (x_off, y_off, scale) = _resize_canvas(result.cv2_image.copy())

    def rp(idx): return _landmark_px(result, scale, x_off, y_off, idx)

    l_sh  = rp(LM.LEFT_SHOULDER);  r_sh  = rp(LM.RIGHT_SHOULDER)
    l_hip = rp(LM.LEFT_HIP);       r_hip = rp(LM.RIGHT_HIP)
    l_kn  = rp(LM.LEFT_KNEE);      r_kn  = rp(LM.RIGHT_KNEE)
    l_ank = rp(LM.LEFT_ANKLE);     r_ank = rp(LM.RIGHT_ANKLE)

    _draw_line(img, l_sh, r_sh, C_SHOULDER, 3)
    _draw_line(img, l_hip, r_hip, C_HIP, 3)

    sh_mid  = _mid(l_sh, r_sh)
    hip_mid = _mid(l_hip, r_hip)
    _draw_dashed(img, sh_mid, hip_mid, (180,180,180), 1, 8)

    # 脊柱參考虛線（不加浮動文字）
    if sh_mid and hip_mid:
        _draw_dashed(img, (sh_mid[0], 0), (hip_mid[0], ANNOTATION_HEIGHT),
                     C_SPINE, 1, 14)

    for p1, p2 in [(l_hip,l_kn),(r_hip,r_kn),(l_kn,l_ank),(r_kn,r_ank)]:
        _draw_line(img, p1, p2, C_GRAY, 1)

    for pt in [l_sh, r_sh, l_hip, r_hip, l_kn, r_kn, l_ank, r_ank]:
        _draw_lm(img, pt)

    # 標籤移至圖例（不在線條上浮動）

    _mark_asymmetry(img, l_sh, r_sh, findings.get("shoulder_asymmetry",{}), "肩")
    _mark_asymmetry(img, l_hip, r_hip, findings.get("pelvic_asymmetry",{}), "髖")

    if result.is_low_confidence():
        _info_banner(img, "背面偵測信心偏低，結果僅供參考")

    _watermark(img, "背面姿勢分析")
    _draw_legend(img, [("肩線", C_SHOULDER), ("髖線", C_HIP), ("脊柱參考", C_SPINE)])
    return img


def annotate_side_view(result: DetectionResult, findings: dict,
                       side: str = "left") -> np.ndarray:
    """
    側面站姿標注：使用 posture_rules 校正後的耳廓 + 肩峰位置繪製。
    校正前（raw）的盂肱關節以小虛圓標示，供對照參考。
    """
    img, (x_off, y_off, scale) = _resize_canvas(result.cv2_image.copy())

    def rp(idx): return _landmark_px(result, scale, x_off, y_off, idx)
    def sp(pt):
        """原圖座標 → 畫布座標"""
        if pt is None: return None
        return (int(pt[0]*scale)+x_off, int(pt[1]*scale)+y_off)

    # ── 取 posture_rules 校正後地標 ──────────────────────
    lm_corr = findings.get("_lm_corrected", {})

    # 耳廓（校正後）
    ear = sp(lm_corr["ear_corrected"]) if lm_corr.get("ear_corrected") else (
          rp(LM.LEFT_EAR) or rp(LM.RIGHT_EAR))

    # 肩峰（估算後）
    sh   = sp(lm_corr["acromion"]) if lm_corr.get("acromion") else (
           rp(LM.LEFT_SHOULDER) or rp(LM.RIGHT_SHOULDER))
    # 原始盂肱關節（標小虛圓對照）
    sh_j = sp(lm_corr.get("shoulder_raw"))

    # 其餘地標
    hip   = sp(lm_corr["hip"])   if lm_corr.get("hip")   else (rp(LM.LEFT_HIP)   or rp(LM.RIGHT_HIP))
    knee  = sp(lm_corr["knee"])  if lm_corr.get("knee")  else (rp(LM.LEFT_KNEE)  or rp(LM.RIGHT_KNEE))
    ankle = sp(lm_corr["ankle"]) if lm_corr.get("ankle") else (rp(LM.LEFT_ANKLE) or rp(LM.RIGHT_ANKLE))
    nose  = rp(LM.NOSE)

    # ── ① 耳垂垂直線 ──
    if ear:
        _draw_dashed(img, (ear[0], max(0, ear[1]-40)),
                     (ear[0], ANNOTATION_HEIGHT), (160,210,255), 1, 12)
        _label(img, "耳垂垂直線", (ear[0]+6, ear[1]+22),
               color_rgb=(160,210,255), size=12)

    # ── ② 頸椎角（耳→肩）──
    _draw_line(img, ear, sh, C_MEASURE, 2)
    cervical = findings.get("cervical_angle", {})
    cervical_angle = cervical.get("angle")
    if ear and sh and cervical_angle is not None:
        mid_neck = _mid(ear, sh)
        if mid_neck:
            _angle_label(img, cervical_angle, "頸椎角",
                         (mid_neck[0]+6, mid_neck[1]-8),
                         color_rgb=PC_MEASURE, size=13)

    # ── ③ FSA 前傾肩角 ──
    fsa     = findings.get("forward_shoulder_angle", {})
    fsa_angle = fsa.get("angle")
    c7_raw  = fsa.get("c7_approx")
    if c7_raw and sh:
        # 將原圖座標轉換為畫布座標
        c7_canvas = (
            int(c7_raw[0] / result.w * (ANNOTATION_WIDTH  - 2*x_off)) + x_off,
            int(c7_raw[1] / result.h * (ANNOTATION_HEIGHT - 2*y_off)) + y_off,
        )
        cv2.circle(img, c7_canvas, 7, C_SHOULDER, -1, cv2.LINE_AA)
        _draw_line(img, c7_canvas, sh, C_SHOULDER, 2)
        # 角度鉛直參考線
        _draw_dashed(img, c7_canvas,
                     (c7_canvas[0], c7_canvas[1]+60), C_SHOULDER, 1, 8)
        if fsa_angle is not None:
            _angle_label(img, fsa_angle, "FSA",
                         (c7_canvas[0]-55, c7_canvas[1]-4),
                         color_rgb=PC_SHOULDER, size=14)
        _label(img, "C7", (c7_canvas[0]+8, c7_canvas[1]-6),
               color_rgb=PC_SHOULDER, size=12)

    # ── ④ 骨盆傾斜（肩→髖）──
    pelvic = findings.get("pelvic_tilt_angle", {})
    pelvic_angle = pelvic.get("angle")
    _draw_line(img, sh, hip, C_HIP, 2)
    if sh and hip and pelvic_angle is not None:
        mid_torso = _mid(sh, hip)
        if mid_torso:
            _angle_label(img, pelvic_angle, "骨盆估算",
                         (mid_torso[0]+6, mid_torso[1]-6),
                         color_rgb=PC_HIP, size=13)

    # ── ⑤ 膝關節角度 ──
    knee_data = findings.get("knee_angle", {})
    knee_angle = knee_data.get("angle")
    _draw_line(img, hip,   knee,  C_GRAY, 2)
    _draw_line(img, knee,  ankle, C_GRAY, 2)
    if knee and knee_angle is not None:
        _angle_arc(img, knee, hip, ankle, radius=30)
        _angle_label(img, knee_angle, "膝角",
                     (knee[0]+10, knee[1]-6),
                     color_rgb=PC_ANGLE, size=13)

    # ── 地標 ──
    for pt in [sh, hip, knee, ankle]:
        _draw_lm(img, pt)
    for pt in [ear, nose]:
        _draw_lm(img, pt, color=(200,200,255), r=4)

    # 原始盂肱關節（虛圓標示，與肩峰做對照）
    if sh_j and sh and sh_j != sh:
        cv2.circle(img, sh_j, 8, (120,120,180), 1, cv2.LINE_AA)   # 虛圓
        # 連線顯示偏移方向
        cv2.line(img, sh_j, sh, (120,120,180), 1, cv2.LINE_AA)

    # 肩峰、髖、踝標示
    if sh:
        _label(img, "肩峰(估算)", (sh[0]+8, sh[1]-20), color_rgb=PC_SHOULDER, size=12)
    if hip:
        _label(img, "髖",   (hip[0]+8, hip[1]-6),   color_rgb=PC_HIP, size=12)
    if ankle:
        _label(img, "踝",   (ankle[0]+8, ankle[1]-6), color_rgb=PC_GREEN, size=12)
    if ear:
        _label(img, "耳廓", (ear[0]+8, ear[1]+6),   color_rgb=(200,200,255), size=12)

    # 頭部前引傾向
    fh = findings.get("forward_head", {})
    if fh.get("grade","") not in ("接近中立","無法評估",""):
        if ear and sh:
            _label(img, f"前引: {fh.get('grade','')}", (ear[0]-70, ear[1]-22),
                   color_rgb=(255,120,80), size=12)

    side_zh = "左側" if side == "left" else "右側"
    _watermark(img, f"{side_zh}姿勢分析")
    _draw_legend(img, [("肩峰", C_SHOULDER), ("髖", C_HIP), ("角度", C_ANGLE)])
    return img


def annotate_forward_bend(result: DetectionResult, findings: dict) -> np.ndarray:
    """前彎測試標注：脊柱中線、肩部對稱參考"""
    img, (x_off, y_off, scale) = _resize_canvas(result.cv2_image.copy())

    def rp(idx): return _landmark_px(result, scale, x_off, y_off, idx)

    l_sh  = rp(LM.LEFT_SHOULDER);  r_sh  = rp(LM.RIGHT_SHOULDER)
    l_hip = rp(LM.LEFT_HIP);       r_hip = rp(LM.RIGHT_HIP)
    l_kn  = rp(LM.LEFT_KNEE);      r_kn  = rp(LM.RIGHT_KNEE)

    sh_mid  = _mid(l_sh, r_sh)
    hip_mid = _mid(l_hip, r_hip)

    # ── 肩線 ──
    _draw_line(img, l_sh, r_sh, C_SHOULDER, 3)

    # ── 脊柱中線 ──
    if sh_mid and hip_mid:
        _draw_line(img, sh_mid, hip_mid, C_SPINE, 3)
        _draw_dashed(img, (sh_mid[0], 0), (sh_mid[0], ANNOTATION_HEIGHT),
                     (170,130,240), 1, 16)
        _label(img, "脊柱中線", (sh_mid[0]+6, sh_mid[1]-22),
               color_rgb=PC_SPINE, size=13)

    # ── 對稱參考虛線 ──
    if sh_mid:
        offset = 70
        _draw_dashed(img, (sh_mid[0]-offset, 0),
                     (sh_mid[0]-offset, ANNOTATION_HEIGHT), (90,90,90), 1, 22)
        _draw_dashed(img, (sh_mid[0]+offset, 0),
                     (sh_mid[0]+offset, ANNOTATION_HEIGHT), (90,90,90), 1, 22)

    for p1, p2 in [(l_hip,l_kn),(r_hip,r_kn)]:
        _draw_line(img, p1, p2, C_GRAY, 1)

    for pt in [l_sh, r_sh, l_hip, r_hip, l_kn, r_kn]:
        _draw_lm(img, pt)

    # ── 偏移狀態標示 ──
    spine_shift = findings.get("spine_shift", {})
    grade = spine_shift.get("grade", "")
    if "偏移" in grade or "不對稱" in grade:
        _info_banner(img, "觀察到脊柱偏移傾向，詳見報告說明", bg_rgb=(0,60,180))
    else:
        _info_banner(img, "前彎測試 — 脊柱對稱性觀察", bg_rgb=(0,80,30))

    # ── 肩部高低差數值 ──
    sh_sym = findings.get("shoulder_symmetry", {})
    if l_sh and r_sh and sh_sym.get("grade","") != "無法評估":
        higher = l_sh if l_sh[1] < r_sh[1] else r_sh
        _label(img, sh_sym.get("grade",""), (higher[0]+10, higher[1]-14),
               color_rgb=(255,180,50), size=12)

    _watermark(img, "背向前彎動作評估")
    return img


# ── 共用：不對稱標示（簡潔設計版）──────────────────────

def _mark_asymmetry(img: np.ndarray, p1, p2, asym: dict, part: str) -> None:
    """
    在偏高側繪製標記。
    設計原則：簡潔、不擁擠、一眼可讀。
    - 偏高側：實心三角箭頭（▲）+ 等級文字
    - 偏低側：空心小點
    - 文字放在畫面左右空白區，避免壓在線條上
    """
    if not asym or p1 is None or p2 is None:
        return
    grade = asym.get("grade", "")
    if grade in ("正常範圍", "無法評估", ""):
        return

    higher = p1 if p1[1] < p2[1] else p2
    lower  = p2 if p1[1] < p2[1] else p1

    # 顏色依嚴重程度
    if "明顯" in grade:
        c_bgr = (0,  70, 255);  c_rgb = (255,  70,  0)
    elif "中度" in grade:
        c_bgr = (0, 140, 255);  c_rgb = (255, 140,  0)
    else:                        # 輕度
        c_bgr = (20, 200, 200); c_rgb = (200, 200,  20)

    # 偏高側：實心圓 + 向上三角形標記
    cv2.circle(img, higher, 10, c_bgr, -1,  cv2.LINE_AA)   # 填實
    cv2.circle(img, higher, 12, c_bgr,  2,  cv2.LINE_AA)   # 外框
    # 向上箭頭（三角形）
    tx, ty = higher[0], higher[1] - 18
    pts = np.array([[tx, ty-9], [tx-7, ty+4], [tx+7, ty+4]], np.int32)
    cv2.fillPoly(img, [pts], c_bgr)

    # 偏低側：小空心圓，表示對比
    cv2.circle(img, lower, 7, c_bgr, 2, cv2.LINE_AA)

    # 文字標籤：放在圖像右側固定區域，避免遮住地標
    img_w = img.shape[1]
    label_x = img_w - 160   # 固定靠右
    label_y = higher[1] - 8

    # 確保不超出圖像邊界
    label_y = max(10, min(label_y, img.shape[0] - 30))

    # 連接線（從地標到標籤）
    cv2.line(img, (higher[0]+12, higher[1]),
             (label_x - 4, label_y + 7), c_bgr, 1, cv2.LINE_AA)

    grade_short = grade.replace("不對稱", "").replace("範圍", "")  # 縮短顯示
    _label(img, f"▲ {part}偏高  {grade_short}",
           (label_x, label_y), color_rgb=c_rgb, size=13,
           bg=(20, 20, 20))
