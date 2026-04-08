# utils/file_utils.py
# ============================================================
# DailyFit Posture Report MVP — 檔案管理工具
# ============================================================

import os
import json
import re
from datetime import datetime
from pathlib import Path

from utils.constants import (
    OUTPUT_ANNOTATED, OUTPUT_REPORTS_HTML, OUTPUT_REPORTS_PDF, OUTPUT_DATA
)


def create_output_dirs() -> None:
    """建立所有輸出資料夾（若不存在）"""
    for path in [OUTPUT_ANNOTATED, OUTPUT_REPORTS_HTML, OUTPUT_REPORTS_PDF, OUTPUT_DATA]:
        Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """將客戶姓名轉為安全的檔名字串"""
    # 僅保留字母、數字、底線、連字號
    safe = re.sub(r"[^\w\-]", "_", name.strip())
    return safe[:30]  # 限制長度


def generate_filename(client_name: str, view: str = "", ext: str = "jpg") -> str:
    """生成帶時間戳記的檔名"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = sanitize_filename(client_name)
    parts = [safe_name, ts]
    if view:
        parts.append(view)
    return "_".join(parts) + f".{ext}"


def save_annotated_image(img_array, client_name: str, view: str) -> str:
    """儲存標注圖像並回傳路徑"""
    import cv2
    create_output_dirs()
    filename = generate_filename(client_name, view, "jpg")
    filepath = os.path.join(OUTPUT_ANNOTATED, filename)
    cv2.imwrite(filepath, img_array)
    return filepath


def save_assessment_json(client_info: dict, analysis_results: dict,
                         score_data: dict) -> str:
    """
    儲存評估 JSON，結構如下：
    {
      "timestamp": ...,
      "client": {...},
      "analysis": {...},
      "score": {...}
    }
    回傳儲存路徑。
    """
    create_output_dirs()
    timestamp = datetime.now().isoformat()
    data = {
        "timestamp": timestamp,
        "client": client_info,
        "analysis": analysis_results,
        "score": score_data,
    }
    filename = generate_filename(client_info.get("name", "unknown"), ext="json")
    filepath = os.path.join(OUTPUT_DATA, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath


def save_html_report(html_content: str, client_name: str) -> str:
    """儲存 HTML 報告並回傳路徑"""
    create_output_dirs()
    filename = generate_filename(client_name, ext="html")
    filepath = os.path.join(OUTPUT_REPORTS_HTML, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    return filepath
