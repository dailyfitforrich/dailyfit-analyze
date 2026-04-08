# analyzer/posture_rules.py
# ============================================================
# DailyFit 姿勢分析規則  v4.1
#
# 主要修正：
#   ① 耳朵地標校正（EAR → 眼後估算，解決鬢角偏移問題）
#   ② 肩峰估算修正（盂肱關節 → 肩峰近似，解決肩位跑掉問題）
#   ③ 側面最佳地標選取（依 visibility 選最可信側）
# ============================================================

import math
import numpy as np
from typing import Optional, Dict, Any, Tuple

from analyzer.pose_detector import DetectionResult, LM
from utils.constants import *


# ═══════════════════════════════════════════════════════════
# 幾何輔助函式
# ═══════════════════════════════════════════════════════════

def calc_angle(p1, p2, p3) -> Optional[float]:
    """計算 p1→p2→p3 的夾角（度）"""
    if any(p is None for p in [p1, p2, p3]):
        return None
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]], float)
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]], float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_a)))


def calc_vertical_angle(p1, p2) -> Optional[float]:
    """p1→p2 連線與鉛直線夾角（度，0–90）"""
    if p1 is None or p2 is None:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return float(math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6)))


def dist(p1, p2) -> float:
    if p1 is None or p2 is None:
        return 0.0
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])


def midpoint(p1, p2) -> Optional[Tuple[int, int]]:
    if p1 is None or p2 is None:
        return None
    return (int((p1[0]+p2[0])//2), int((p1[1]+p2[1])//2))


def _vis(result: DetectionResult, idx: int) -> float:
    """取得地標可見度（0–1），地標不存在時回傳 0"""
    if not result.detected or result.landmarks is None:
        return 0.0
    lms = result.landmarks.landmark
    if idx >= len(lms):
        return 0.0
    return lms[idx].visibility


def _raw_px(result: DetectionResult, idx: int) -> Optional[Tuple[int, int]]:
    """
    不過 visibility 門檻直接取像素座標。
    用於需要低信心地標做幾何推算的情況。
    """
    if not result.detected or result.landmarks is None:
        return None
    lms = result.landmarks.landmark
    if idx >= len(lms):
        return None
    lm = lms[idx]
    return (int(lm.x * result.w), int(lm.y * result.h))


# ═══════════════════════════════════════════════════════════
# ① 側面地標提取（含耳朵校正 + 肩峰估算）
# ═══════════════════════════════════════════════════════════

def _select_side_landmarks(result: DetectionResult, side: str) -> Dict:
    """
    為側面分析選取最可信的地標，並回傳校正後的位置。
    
    side = "left"  → 受測者左側面對鏡頭
    side = "right" → 受測者右側面對鏡頭

    回傳 dict 包含：
        ear_corrected   : 校正後耳朵位置
        acromion        : 估算肩峰位置
        shoulder_raw    : 原始盂肱關節位置（保留供除錯）
        hip, knee, ankle: 標準地標
        debug           : 除錯資訊字串
    """
    # ── 決定近側（面向鏡頭的那側）────────────────────────
    # 左側照：左側靠近鏡頭，左側地標 visibility 通常較高
    # 右側照：右側靠近鏡頭
    if side == "left":
        near_ear  = LM.LEFT_EAR;   far_ear  = LM.RIGHT_EAR
        near_eye  = LM.LEFT_EYE;   far_eye  = LM.RIGHT_EYE
        near_sh   = LM.LEFT_SHOULDER; far_sh = LM.RIGHT_SHOULDER
        near_elb  = LM.LEFT_ELBOW; far_elb  = LM.RIGHT_ELBOW
        near_hip  = LM.LEFT_HIP;   far_hip  = LM.RIGHT_HIP
        near_kn   = LM.LEFT_KNEE;  far_kn   = LM.RIGHT_KNEE
        near_ank  = LM.LEFT_ANKLE; far_ank  = LM.RIGHT_ANKLE
    else:
        near_ear  = LM.RIGHT_EAR;  far_ear  = LM.LEFT_EAR
        near_eye  = LM.RIGHT_EYE;  far_eye  = LM.LEFT_EYE
        near_sh   = LM.RIGHT_SHOULDER; far_sh = LM.LEFT_SHOULDER
        near_elb  = LM.RIGHT_ELBOW; far_elb = LM.LEFT_ELBOW
        near_hip  = LM.RIGHT_HIP;  far_hip  = LM.LEFT_HIP
        near_kn   = LM.RIGHT_KNEE; far_kn   = LM.LEFT_KNEE
        near_ank  = LM.RIGHT_ANKLE; far_ank = LM.LEFT_ANKLE

    # ── 選取肩、髖、膝、踝（近側優先）──────────────────
    sh_vis   = _vis(result, near_sh)
    sh_raw   = result.px(near_sh) if sh_vis >= 0.3 else result.px(far_sh)

    hip_vis  = _vis(result, near_hip)
    hip      = result.px(near_hip) if hip_vis >= 0.3 else result.px(far_hip)

    kn_vis   = _vis(result, near_kn)
    knee     = result.px(near_kn) if kn_vis >= 0.3 else result.px(far_kn)

    ank_vis  = _vis(result, near_ank)
    ankle    = result.px(near_ank) if ank_vis >= 0.3 else result.px(far_ank)

    # ── ② 肩峰估算（修正盂肱關節偏移）────────────────────
    # MediaPipe 標記的是盂肱關節（肩關節）
    # 實際肩峰（acromion）在盂肱關節前方、稍微上方
    # 估算方式：
    #   · 取肩→肘方向向量（代表手臂走向）
    #   · 肩峰 ≈ 盂肱關節 + 手臂方向的反向 × 10-15% 手臂長度
    #   · 若肘部不可見：用 shoulder → neck 方向估算
    elbow_raw = (result.px(near_elb)
                 if _vis(result, near_elb) >= 0.3
                 else result.px(far_elb))

    acromion  = _estimate_acromion(sh_raw, elbow_raw, hip)

    # ── ① 耳朵校正（解決鬢角偏移）────────────────────────
    # 問題根源：MediaPipe EAR 地標在側面照常偏向太陽穴/鬢角
    # 校正策略：
    #   1. 取眼睛地標（外眼角，通常在側面很穩定）
    #   2. 從眼後方推算耳廓（tragus）位置
    #   3. 與原始 EAR 地標做加權合成
    ear_raw  = _raw_px(result, near_ear)   # 不卡 visibility，先取原始值
    eye_raw  = _raw_px(result, near_eye)
    nose_raw = _raw_px(result, LM.NOSE)

    ear_corrected, ear_debug = _correct_ear_landmark(
        ear_raw=ear_raw,
        eye_px=eye_raw,
        nose_px=nose_raw,
        shoulder_px=sh_raw,
        ear_visibility=_vis(result, near_ear),
        side=side,
        img_w=result.w,
        img_h=result.h,
    )

    debug_str = (
        f"ear_raw={ear_raw} ear_corrected={ear_corrected} "
        f"sh_raw={sh_raw} acromion={acromion} "
        f"ear_vis={_vis(result, near_ear):.2f} sh_vis={sh_vis:.2f} | "
        f"{ear_debug}"
    )

    return {
        "ear_corrected": ear_corrected,
        "acromion":      acromion,
        "shoulder_raw":  sh_raw,
        "hip":   hip,
        "knee":  knee,
        "ankle": ankle,
        "debug": debug_str,
    }


def _correct_ear_landmark(
    ear_raw:        Optional[Tuple[int, int]],
    eye_px:         Optional[Tuple[int, int]],
    nose_px:        Optional[Tuple[int, int]],
    shoulder_px:    Optional[Tuple[int, int]],
    ear_visibility: float,
    side:           str,
    img_w:          int,
    img_h:          int,
) -> Tuple[Optional[Tuple[int, int]], str]:
    """
    耳朵地標校正核心函式。
    
    MediaPipe EAR 地標在側面照中常出現在鬢角/太陽穴，
    本函式用眼睛位置重新估算耳廓（tragus）的合理位置。

    解剖學依據：
      - 耳廓大約在眼外眥後方 1.5 個眼寬、同高或略低
      - 在側面 2D 投影中，耳應在眼後方且與眼同高或略低 5–8%頭高

    回傳 (校正後位置, 除錯說明字串)
    """
    # ── 情況 A：無眼睛地標 → 只能用原始耳朵 ──
    if eye_px is None:
        return ear_raw, "no_eye_fallback"

    # ── 估算頭部參考尺寸 ──
    # 用眼→肩距離估算頭高（側面照通常清楚）
    if shoulder_px is not None:
        head_ref = dist(eye_px, shoulder_px) * 0.35
    else:
        # fallback：圖像高度 10%
        head_ref = img_h * 0.10
    head_ref = max(head_ref, 15)   # 最小 15px 避免極端值

    # ── 從眼睛位置推算耳廓預期位置 ──
    # 方向：
    #   left side  → 人臉朝右 → 耳在眼睛右方（x 增加）
    #   right side → 人臉朝左 → 耳在眼睛左方（x 減少）
    ear_offset_x = +head_ref * 1.2 if side == "left" else -head_ref * 1.2
    ear_offset_y = +head_ref * 0.05   # 略低於眼睛

    # 若有鼻子地標，可更準確判斷臉朝向
    if nose_px is not None:
        # nose 相對 eye 的方向代表臉朝向
        face_dx = nose_px[0] - eye_px[0]
        # 耳在臉朝向的「反方向」
        if abs(face_dx) > 5:
            # face_dx > 0 → 臉朝右 → 耳在眼睛左方（side=right）
            # face_dx < 0 → 臉朝左 → 耳在眼睛右方（side=left）
            ear_offset_x = -face_dx * 1.5

    eye_based_ear = (
        int(eye_px[0] + ear_offset_x),
        int(eye_px[1] + ear_offset_y),
    )

    # ── 情況 B：無原始耳朵地標 → 完全用眼睛估算 ──
    if ear_raw is None:
        return eye_based_ear, "eye_estimated_no_raw"

    # ── 情況 C：原始耳朵地標信心低 → 偏重眼睛估算 ──
    # 鬢角偏移判斷：原始 EAR 是否太靠近眼睛
    ear_eye_dist = dist(ear_raw, eye_px)
    expected_dist = abs(ear_offset_x) * 0.7   # 最小合理耳眼距

    if ear_visibility < 0.5 or ear_eye_dist < expected_dist:
        # 信心低或距離太近（在鬢角附近）→ 用眼估算
        blend_weight = 0.2 if ear_visibility >= 0.4 else 0.0
        cx = int(eye_based_ear[0] * (1-blend_weight) + ear_raw[0] * blend_weight)
        cy = int(eye_based_ear[1] * (1-blend_weight) + ear_raw[1] * blend_weight)
        return (cx, cy), f"eye_dominant(vis={ear_visibility:.2f},dist={ear_eye_dist:.0f}<{expected_dist:.0f})"

    # ── 情況 D：原始耳朵地標信心高且位置合理 ──
    # 做輕度加權：80% 原始，20% 眼估算（平滑化）
    wx = int(ear_raw[0] * 0.8 + eye_based_ear[0] * 0.2)
    wy = int(ear_raw[1] * 0.8 + eye_based_ear[1] * 0.2)
    return (wx, wy), f"blended_good(vis={ear_visibility:.2f})"


def _estimate_acromion(
    shoulder_px:  Optional[Tuple[int, int]],
    elbow_px:     Optional[Tuple[int, int]],
    hip_px:       Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """
    從盂肱關節估算肩峰（acromion）位置。

    解剖學依據：
      - 肩峰是肩胛骨的外側端，位於盂肱關節前上方
      - 在側面 2D 影像中，肩峰相對盂肱關節的偏移：
          X（前後）: 約為肩肘距離的 8–12% 向「前」方
          Y（上下）: 約為肩肘距離的 3–5% 向「上」方

    估算方式（有肘部時）：
      - 計算肩→肘向量（代表上臂方向）
      - 肩峰 = 盂肱關節 + 垂直於上臂方向的前上方偏移

    估算方式（無肘部時）：
      - 用肩→髖方向的垂直方向估算
    """
    if shoulder_px is None:
        return None

    # ── 有肘部 → 用上臂方向估算 ──────────────────────────
    if elbow_px is not None:
        arm_len = dist(shoulder_px, elbow_px)
        if arm_len < 5:
            arm_len = 5

        # 上臂方向向量（肩→肘）
        arm_dx = elbow_px[0] - shoulder_px[0]
        arm_dy = elbow_px[1] - shoulder_px[1]
        arm_len_actual = math.hypot(arm_dx, arm_dy)

        if arm_len_actual > 0:
            # 正規化
            ux = arm_dx / arm_len_actual
            uy = arm_dy / arm_len_actual
            # 垂直方向（旋轉 90°）→ 代表「肩前方」
            perp_x = -uy
            perp_y =  ux

            # 偏移量：上臂長度的 10%
            offset = arm_len * 0.10

            # 肩峰 = 盂肱關節 + 前方偏移 - 少量上移
            acromion_x = int(shoulder_px[0] + perp_x * offset)
            acromion_y = int(shoulder_px[1] + perp_y * offset - arm_len * 0.04)
            return (acromion_x, acromion_y)

    # ── 無肘部 → 用軀幹方向估算 ──────────────────────────
    if hip_px is not None:
        torso_len = dist(shoulder_px, hip_px)
        if torso_len < 10:
            return shoulder_px

        # 軀幹方向向量（肩→髖）
        tx = hip_px[0] - shoulder_px[0]
        ty = hip_px[1] - shoulder_px[1]
        tlen = math.hypot(tx, ty)

        # 垂直於軀幹（代表側前方）
        px = -ty / tlen
        py =  tx / tlen

        offset = torso_len * 0.06
        acromion_x = int(shoulder_px[0] + px * offset)
        acromion_y = int(shoulder_px[1] - torso_len * 0.02)  # 微上移
        return (acromion_x, acromion_y)

    # ── 無任何參考 → 回傳原始肩關節位置 ──
    return shoulder_px


# ═══════════════════════════════════════════════════════════
# 正面分析
# ═══════════════════════════════════════════════════════════

def analyze_front_view(result: DetectionResult) -> Dict[str, Any]:
    """正面站姿：肩部不對稱、骨盆不對稱、整體平衡"""
    findings = {}
    if not result.detected:
        return {"error": "未偵測到姿勢地標，請確認照片清晰且全身可見。"}

    l_sh  = result.px(LM.LEFT_SHOULDER)
    r_sh  = result.px(LM.RIGHT_SHOULDER)
    l_hip = result.px(LM.LEFT_HIP)
    r_hip = result.px(LM.RIGHT_HIP)

    findings["shoulder_asymmetry"] = _shoulder_asymmetry(l_sh, r_sh, result.w)
    findings["pelvic_asymmetry"]   = _pelvic_asymmetry(l_hip, r_hip, result.w)
    findings["balance_note"]       = _overall_balance(
        findings["shoulder_asymmetry"], findings["pelvic_asymmetry"])

    if result.is_low_confidence():
        findings["confidence_note"] = result.confidence_note()

    return findings


def _shoulder_asymmetry(l_sh, r_sh, img_w: int) -> Dict:
    if l_sh is None or r_sh is None:
        return {"grade":"無法評估","description":"肩部地標信心不足，無法評估。",
                "deduction":0,"dy_norm":None}
    shoulder_width = dist(l_sh, r_sh)
    dy = abs(l_sh[1] - r_sh[1])
    dy_norm = dy / (shoulder_width + 1e-6)

    grade = _asym_grade(dy_norm, SHOULDER_ASYMMETRY_MILD,
                        SHOULDER_ASYMMETRY_MODERATE, SHOULDER_ASYMMETRY_SIGNIFICANT)
    side = "左肩偏高" if l_sh[1] < r_sh[1] else "右肩偏高"

    if grade == "正常範圍":
        return {"grade":"正常範圍",
                "description":"肩部左右高度接近對稱，未觀察到明顯高低差異。",
                "deduction":0,"dy_norm":round(dy_norm,4),"side":""}
    descs = {
        "輕度": f"觀察到輕度肩部高低差異（{side}），可能與習慣性姿勢或慣用手有關。",
        "中度": f"觀察到中度肩部高低不對稱（{side}），建議關注上斜方肌與肩帶使用習慣。",
        "明顯": f"觀察到明顯肩部高低不對稱（{side}），建議進一步評估肩帶與脊柱對位。",
    }
    deductions = {"輕度": SCORE_DEDUCTION_MILD, "中度": SCORE_DEDUCTION_MODERATE,
                  "明顯": SCORE_DEDUCTION_SIGNIFICANT}
    return {"grade":grade,"description":descs[grade],"deduction":deductions[grade],
            "dy_norm":round(dy_norm,4),"side":side}


def _pelvic_asymmetry(l_hip, r_hip, img_w: int) -> Dict:
    if l_hip is None or r_hip is None:
        return {"grade":"無法評估","description":"骨盆地標信心不足，無法評估。",
                "deduction":0,"dy_norm":None}
    hip_width = dist(l_hip, r_hip)
    dy = abs(l_hip[1] - r_hip[1])
    dy_norm = dy / (hip_width + 1e-6)

    grade = _asym_grade(dy_norm, PELVIC_ASYMMETRY_MILD,
                        PELVIC_ASYMMETRY_MODERATE, PELVIC_ASYMMETRY_SIGNIFICANT)
    side = "左側骨盆偏高" if l_hip[1] < r_hip[1] else "右側骨盆偏高"

    if grade == "正常範圍":
        return {"grade":"正常範圍",
                "description":"骨盆左右高度接近對稱，未觀察到明顯骨盆傾斜傾向。",
                "deduction":0,"dy_norm":round(dy_norm,4),"side":""}
    descs = {
        "輕度": f"觀察到輕度骨盆高低差異（{side}），可能與單側使用習慣或髖部緊繃有關。",
        "中度": f"觀察到中度骨盆高低不對稱（{side}），建議關注髖部活動度與核心控制。",
        "明顯": f"觀察到明顯骨盆高低不對稱（{side}），建議系統性評估下肢對位與骨盆穩定性。",
    }
    deductions = {"輕度": SCORE_DEDUCTION_MILD, "中度": SCORE_DEDUCTION_MODERATE,
                  "明顯": SCORE_DEDUCTION_SIGNIFICANT}
    return {"grade":grade,"description":descs[grade],"deduction":deductions[grade],
            "dy_norm":round(dy_norm,4),"side":side}


def _asym_grade(ratio, mild, moderate, significant) -> str:
    if ratio >= significant: return "明顯"
    if ratio >= moderate:    return "中度"
    if ratio >= mild:        return "輕度"
    return "正常範圍"


def _overall_balance(sh_asym: Dict, hip_asym: Dict) -> str:
    grades = {sh_asym.get("grade",""), hip_asym.get("grade","")}
    if "明顯" in grades:
        return "整體正面觀察顯示明顯左右不對稱傾向，肩部與骨盆均有偏差，建議優先改善對稱性訓練。"
    elif "中度" in grades:
        return "正面觀察顯示中度左右不對稱傾向，肩部或骨盆存在偏差，建議針對性加強穩定訓練。"
    elif "輕度" in grades:
        return "正面觀察顯示輕度左右不對稱，整體尚在可接受範圍，建議持續關注並矯正。"
    return "正面觀察整體左右對稱性良好，未見明顯偏差。"


# ═══════════════════════════════════════════════════════════
# 背面分析
# ═══════════════════════════════════════════════════════════

BACK_VIEW_NOTE = "背面偵測地標信心有限，背面分析結果主要用於確認正面觀察，建議以正面與側面為主要參考依據。"

def analyze_back_view(result: DetectionResult) -> Dict[str, Any]:
    """背面站姿：確認肩/骨盆不對稱"""
    if not result.detected:
        return {"error":"背面未偵測到姿勢地標。","limit_note":BACK_VIEW_NOTE}

    findings = {}
    l_sh  = result.px(LM.LEFT_SHOULDER)
    r_sh  = result.px(LM.RIGHT_SHOULDER)
    l_hip = result.px(LM.LEFT_HIP)
    r_hip = result.px(LM.RIGHT_HIP)

    findings["shoulder_asymmetry"] = _shoulder_asymmetry(l_sh, r_sh, result.w)
    findings["pelvic_asymmetry"]   = _pelvic_asymmetry(l_hip, r_hip, result.w)
    findings["limit_note"]         = BACK_VIEW_NOTE

    if result.is_low_confidence():
        findings["confidence_note"] = result.confidence_note()
    return findings


# ═══════════════════════════════════════════════════════════
# 側面分析（使用校正後的地標）
# ═══════════════════════════════════════════════════════════

def analyze_side_view(result: DetectionResult, side: str = "left") -> Dict[str, Any]:
    """
    側面站姿分析。
    使用 _select_side_landmarks() 取得校正後的耳朵與肩峰位置，
    再進行各項角度計算。
    """
    findings = {"side": side}
    if not result.detected:
        return {"error": f"{'左' if side=='left' else '右'}側面未偵測到姿勢地標。",
                "side": side}

    # ── 取得校正後地標 ────────────────────────────────────
    lm = _select_side_landmarks(result, side)
    ear     = lm["ear_corrected"]   # 校正耳廓位置
    sh      = lm["acromion"]        # 估算肩峰位置
    sh_raw  = lm["shoulder_raw"]    # 原始盂肱關節（供除錯用）
    hip     = lm["hip"]
    knee    = lm["knee"]
    ankle   = lm["ankle"]

    # ── 各項分析 ──────────────────────────────────────────
    findings["cervical_angle"]         = _cervical_angle(ear, sh)
    findings["forward_shoulder_angle"] = _forward_shoulder_angle(ear, sh, hip)
    findings["pelvic_tilt_angle"]      = _pelvic_tilt_angle(sh_raw, hip, knee)  # 軀幹傾斜用原始肩
    findings["knee_angle"]             = _knee_angle(hip, knee, ankle)
    findings["forward_head"]           = _forward_head_tendency(ear, sh, hip)

    # 暴露校正後地標給 annotator 使用
    findings["_lm_corrected"] = lm

    if result.is_low_confidence():
        findings["confidence_note"] = result.confidence_note()

    return findings


# ── 各角度計算函式 ────────────────────────────────────────

def _cervical_angle(ear, shoulder) -> Dict:
    """頸椎前傾角：校正耳→肩峰連線與鉛直線夾角"""
    angle = calc_vertical_angle(shoulder, ear)
    if angle is None:
        return {"angle":None,"grade":"無法評估",
                "description":"地標信心不足，無法計算頸椎角度。","deduction":0}

    if angle < CERVICAL_ANGLE_NORMAL:
        return {"angle":round(angle,1),"grade":"接近中立",
                "description":f"頸椎傾斜角約 {angle:.1f}°，頸部排列接近中立位。",
                "deduction":0}
    elif angle <= CERVICAL_ANGLE_MILD:
        return {"angle":round(angle,1),"grade":"輕度前傾",
                "description":f"頸椎傾斜角約 {angle:.1f}°，顯示輕度頭部前引姿勢傾向。",
                "deduction":SCORE_DEDUCTION_MILD}
    else:
        return {"angle":round(angle,1),"grade":"明顯前傾",
                "description":f"頸椎傾斜角約 {angle:.1f}°，顯示明顯頭部前引傾向，可能與長時間低頭使用3C或胸椎活動度不足有關。",
                "deduction":SCORE_DEDUCTION_MODERATE}


def _forward_shoulder_angle(ear, acromion, hip) -> Dict:
    """
    前傾肩角 FSA：C7近似→肩峰→鉛直線夾角
    注意：此處 acromion 已是校正後的肩峰估算位置。
    """
    if ear is None or acromion is None:
        return {"angle":None,"grade":"無法評估",
                "description":"地標信心不足，無法計算前傾肩角。","deduction":0}

    # C7 近似：耳廓下方、介於耳與肩之間的頸椎位置
    if hip is not None:
        neck_x = int(ear[0] + (acromion[0] - ear[0]) * 0.25)
        neck_y = int(acromion[1] - abs(acromion[1] - ear[1]) * 0.12)
    else:
        neck_x, neck_y = ear[0], acromion[1] - 8
    c7_approx = (neck_x, neck_y)

    angle = calc_vertical_angle(c7_approx, acromion)
    if angle is None:
        return {"angle":None,"grade":"無法評估",
                "description":"無法計算前傾肩角。","deduction":0}

    if angle < FSA_THRESHOLD:
        grade = "正常或輕微"
        desc  = f"前傾肩角（FSA）約 {angle:.1f}°，肩部位置接近正常對位。"
        deduction = 0
    else:
        grade = "明顯圓肩傾向"
        desc  = (f"前傾肩角（FSA）約 {angle:.1f}°，顯示肩部前移傾向，"
                 "可能呈現圓肩姿勢。這通常與胸肌緊繃及上背控制不足有關。")
        deduction = SCORE_DEDUCTION_MODERATE

    return {"angle":round(angle,1),"grade":grade,"description":desc,
            "deduction":deduction,"c7_approx":c7_approx}


def _pelvic_tilt_angle(shoulder, hip, knee) -> Dict:
    """骨盆傾斜角：軀幹線（肩→髖）與鉛直線夾角（使用原始盂肱關節）"""
    if hip is None or shoulder is None:
        return {"angle":None,"grade":"無法評估",
                "description":"地標信心不足，無法估算骨盆傾斜角。","deduction":0}

    angle = calc_vertical_angle(shoulder, hip)
    if angle is None:
        return {"angle":None,"grade":"無法評估",
                "description":"無法計算骨盆傾斜角。","deduction":0}

    if angle <= 2:
        return {"angle":round(angle,1),"grade":"接近中立",
                "description":f"骨盆傾斜角估約 {angle:.1f}°，骨盆位置觀察接近中立。",
                "deduction":0,
                "limit_note":"靜態 2D 估算，具固有限制，僅供教練參考。"}
    elif angle <= 8:
        return {"angle":round(angle,1),"grade":"輕度前傾傾向",
                "description":f"骨盆傾斜角估約 {angle:.1f}°，側面觀察顯示可能存在輕度骨盆前傾傾向。",
                "deduction":SCORE_DEDUCTION_MILD,
                "limit_note":"靜態 2D 估算，具固有限制，僅供教練參考。"}
    else:
        return {"angle":round(angle,1),"grade":"明顯前傾傾向",
                "description":f"骨盆傾斜角估約 {angle:.1f}°，側面觀察顯示骨盆前傾傾向明顯，可能伴隨腰椎過度前凸。",
                "deduction":SCORE_DEDUCTION_MODERATE,
                "limit_note":"靜態 2D 估算，具固有限制，僅供教練參考。"}


def _knee_angle(hip, knee, ankle) -> Dict:
    """膝關節角度：髖→膝→踝夾角"""
    if hip is None or knee is None or ankle is None:
        return {"angle":None,"grade":"無法評估",
                "description":"地標信心不足，無法計算膝關節角度。","deduction":0}

    angle = calc_angle(hip, knee, ankle)
    if angle is None:
        return {"angle":None,"grade":"無法評估",
                "description":"無法計算膝關節角度。","deduction":0}

    if angle >= KNEE_HYPEREXTENSION_THRESHOLD:
        return {"angle":round(angle,1),"grade":"接近正常",
                "description":f"膝關節角度約 {angle:.1f}°，未觀察到明顯膝過伸傾向。",
                "deduction":0}
    else:
        return {"angle":round(angle,1),"grade":"疑似過伸傾向",
                "description":f"膝關節角度約 {angle:.1f}°，側面觀察顯示可能的膝關節過伸傾向，建議加強膝關節控制訓練。",
                "deduction":SCORE_DEDUCTION_MILD}


def _forward_head_tendency(ear, shoulder, hip) -> Dict:
    """頭部前引：耳相對肩峰的水平偏移（使用校正後耳廓）"""
    if ear is None or shoulder is None:
        return {"grade":"無法評估","description":"地標信心不足，無法評估頭部前引。",
                "deduction":0,"offset_ratio":None}

    h_offset = shoulder[0] - ear[0]
    torso_ref = abs(shoulder[1] - (hip[1] if hip else shoulder[1]-200)) or 200
    ratio = h_offset / (torso_ref + 1e-6)

    if abs(ratio) < FORWARD_HEAD_MILD_RATIO:
        return {"grade":"接近中立","description":"頭部位置觀察接近中立，未顯示明顯前引傾向。",
                "deduction":0,"offset_ratio":round(ratio,4)}
    elif abs(ratio) < FORWARD_HEAD_MODERATE_RATIO:
        return {"grade":"輕度前引","description":"側面觀察顯示輕度頭部前引傾向，耳廓位置略微偏前於肩峰。",
                "deduction":SCORE_DEDUCTION_MILD,"offset_ratio":round(ratio,4)}
    else:
        return {"grade":"明顯前引","description":"側面觀察顯示明顯頭部前引傾向，頸椎可能承受較大壓力，建議改善頸部深層肌肉控制。",
                "deduction":SCORE_DEDUCTION_MODERATE,"offset_ratio":round(ratio,4)}


# ═══════════════════════════════════════════════════════════
# 彙整所有分析結果
# ═══════════════════════════════════════════════════════════

def aggregate_findings(front, back, left_side, right_side,
                       bend=None) -> Dict[str, Any]:
    """彙整四面結果，提取前 3 項觀察與總扣分。"""
    all_issues = []

    def collect(src, label_prefix=""):
        if not src or "error" in src:
            return
        for key, val in src.items():
            if key.startswith("_"):   # 跳過內部欄位
                continue
            if isinstance(val, dict) and "description" in val:
                deduction = val.get("deduction", 0)
                grade     = val.get("grade", "")
                if grade not in ("正常範圍","接近中立","接近正常","對稱",
                                 "未觀察到明顯偏移","正常或輕微","無法評估",""):
                    all_issues.append({
                        "key": key, "description": val["description"],
                        "deduction": deduction, "grade": grade,
                        "source": label_prefix,
                    })

    collect(front,      "正面")
    collect(back,       "背面")
    collect(left_side,  "左側")
    collect(right_side, "右側")

    all_issues.sort(key=lambda x: x["deduction"], reverse=True)
    top3 = all_issues[:3]
    total_deduction = min(sum(i["deduction"] for i in all_issues), 100)

    return {"all_issues": all_issues, "top3": top3, "total_deduction": total_deduction}
