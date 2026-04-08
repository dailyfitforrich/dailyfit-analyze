# analyzer/scoring.py
# ============================================================
# DailyFit Posture Report MVP — 計分系統
# 這是健身教練內部參考分數，非醫療評分
# ============================================================

from typing import Dict, Any, List, Tuple
from utils.text_templates import get_score_label


def calculate_score(aggregated: Dict[str, Any]) -> Dict[str, Any]:
    """
    計算整體姿勢平衡分數。
    
    從 100 分開始，依每個問題嚴重程度扣分。
    回傳：score, category, label, description, score_breakdown
    """
    total_deduction = aggregated.get("total_deduction", 0)
    score = max(0, 100 - total_deduction)

    label, category_desc = get_score_label(score)

    # 計分明細
    breakdown = []
    for issue in aggregated.get("all_issues", []):
        if issue["deduction"] > 0:
            breakdown.append({
                "source": issue["source"],
                "description": _short_desc(issue["description"]),
                "deduction": issue["deduction"],
                "grade": issue["grade"],
            })

    return {
        "score": score,
        "label": label,
        "category_description": category_desc,
        "total_deduction": total_deduction,
        "breakdown": breakdown,
        "note": "本分數為健身教練內部參考指標，非醫療評分。",
    }


def get_top_observations(aggregated: Dict[str, Any]) -> List[Dict]:
    """回傳前 3 項最值得關注的觀察（依扣分排序）"""
    return aggregated.get("top3", [])


def _short_desc(desc: str, max_len: int = 50) -> str:
    """截短說明文字"""
    return desc if len(desc) <= max_len else desc[:max_len] + "..."


def get_coaching_suggestions(aggregated: Dict[str, Any]) -> List[str]:
    """
    依分析結果生成訓練建議清單。
    """
    from utils.text_templates import COACHING_SUGGESTIONS, COACHING_GENERAL

    suggestions = []
    seen_keys = set()

    for issue in aggregated.get("all_issues", []):
        key = issue.get("key", "")
        # 映射各 key 至建議
        mapping = {
            "forward_head":            "forward_head",
            "forward_shoulder_angle":  "rounded_shoulder",
            "cervical_angle":          "forward_head",
            "pelvic_tilt_angle":       "anterior_pelvic_tilt",
            "shoulder_asymmetry":      "shoulder_asymmetry",
            "pelvic_asymmetry":        "pelvic_asymmetry",
            "knee_angle":              "knee_hyperextension",
            "shoulder_symmetry":       "scoliosis_tendency",
            "spine_shift":             "scoliosis_tendency",
        }
        suggestion_key = mapping.get(key)
        if suggestion_key and suggestion_key not in seen_keys:
            text = COACHING_SUGGESTIONS.get(suggestion_key)
            if text:
                suggestions.append(text)
                seen_keys.add(suggestion_key)

    suggestions.extend(COACHING_GENERAL)
    return suggestions
