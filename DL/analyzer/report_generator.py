# analyzer/report_generator.py
# ============================================================
# DailyFit Posture Report MVP — HTML 報告生成器
# 生成教練與客戶可直接閱讀的體態分析報告
# ============================================================

from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import os
import base64

from utils.image_utils import image_to_base64, pil_to_base64
from utils.text_templates import DISCLAIMER, STATIC_ANALYSIS_LIMIT, get_score_label
from analyzer.scoring import get_coaching_suggestions


def _load_logo_b64() -> str:
    """載入 Logo 為 base64，找不到時回傳空字串"""
    # 優先使用較小的 JPEG 版（白底），供 HTML 報告嵌入
    for fname in ("logo_report.jpg", "logo.png"):
        logo_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "assets", fname)
        )
        if os.path.exists(logo_path):
            mime = "jpeg" if fname.endswith(".jpg") else "png"
            with open(logo_path, "rb") as f:
                return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()
    return ""


def generate_html_report(
    client_info: Dict[str, Any],
    front_findings: Dict,
    back_findings: Dict,
    left_findings: Dict,
    right_findings: Dict,
    bend_findings: Optional[Dict],
    aggregated: Dict,
    score_data: Dict,
    annotated_images: Dict[str, Optional[np.ndarray]],
    original_images: Dict,
) -> str:
    """
    生成完整 HTML 報告。
    annotated_images: {'front': ndarray, 'back': ndarray, 'left': ndarray,
                       'right': ndarray, 'bend': ndarray or None}
    """
    coaching = get_coaching_suggestions(aggregated)
    score = score_data["score"]
    label, _ = get_score_label(score)
    score_color = _score_color(score)
    ts = datetime.now().strftime("%Y 年 %m 月 %d 日 %H:%M")
    logo_b64 = _load_logo_b64()
    logo_html = (
        f'<img src="{logo_b64}" style="height:52px;width:auto;object-fit:contain;" alt="Logo">'
        if logo_b64 else ""
    )

    # 嵌入圖像 base64
    def img_b64(key):
        arr = annotated_images.get(key)
        if arr is None:
            return ""
        return image_to_base64(arr)

    front_img = img_b64("front")
    back_img  = img_b64("back")
    left_img  = img_b64("left")
    right_img = img_b64("right")

    top3 = aggregated.get("top3", [])

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DailyFit健身空間 體態分析報告 — {client_info.get('name','')}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');
  
  :root {{
    --bg:      #F8FAFC;
    --surface: #FFFFFF;
    --card:    #F1F5F9;
    --accent:  #0F52BA;
    --accent2: #0EA5E9;
    --gold:    #D97706;
    --danger:  #DC2626;
    --text:    #0F172A;
    --muted:   #475569;
    --border:  #E2E8F0;
    --green:   #16A34A;
    --green-lt:#DCFCE7;
    --amber:   #D97706;
    --amber-lt:#FEF3C7;
    --red:     #DC2626;
    --red-lt:  #FEE2E2;
    --blue-lt: #EFF6FF;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.7;
  }}

  .report-wrap {{ max-width: 860px; margin: 0 auto; padding: 24px 16px 60px; }}

  /* ── Header ── */
  .report-header {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .brand {{ font-size: 20px; font-weight: 700; color: var(--text); letter-spacing: -0.3px; }}
  .brand span {{ color: var(--accent); }}
  .report-meta {{ text-align: right; color: var(--muted); font-size: 12px; }}
  .report-meta .client-name {{
    font-size: 18px; color: var(--text);
    font-weight: 700; margin-bottom: 4px;
  }}

  /* ── Score Card ── */
  .score-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .score-circle {{
    width: 100px; height: 100px;
    border-radius: 50%;
    border: 5px solid {score_color};
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    flex-shrink: 0;
    background: {score_color}0D;
  }}
  .score-num {{ font-size: 34px; font-weight: 700; color: {score_color}; line-height: 1; }}
  .score-unit {{ font-size: 11px; color: var(--muted); margin-top: 2px; }}
  .score-detail {{ flex: 1; }}
  .score-label {{
    display: inline-block;
    background: {score_color}18;
    color: {score_color};
    border: 1px solid {score_color}44;
    border-radius: 6px;
    padding: 4px 14px;
    font-size: 14px; font-weight: 700;
    margin-bottom: 8px;
  }}
  .score-note {{ color: var(--muted); font-size: 11px; margin-top: 8px; }}

  /* ── Top 3 Issues ── */
  .top3-list {{ display: flex; flex-direction: column; gap: 6px; margin-top: 10px; }}
  .top3-item {{
    background: var(--blue-lt);
    border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0;
    padding: 8px 12px;
    font-size: 12px;
    color: var(--text);
  }}
  .top3-item .issue-source {{
    font-size: 10px; font-weight: 700;
    color: var(--accent); letter-spacing: 0.5px;
    text-transform: uppercase; margin-bottom: 2px;
  }}

  /* ── Section ── */
  .section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  }}
  .section-title {{
    font-size: 11px; font-weight: 700;
    color: var(--accent);
    letter-spacing: 1px; text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
    margin-bottom: 16px;
  }}

  /* ── Client Info Grid ── */
  .info-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }}
  .info-item {{
    background: var(--card);
    border-radius: 8px;
    padding: 10px 14px;
    border: 1px solid var(--border);
  }}
  .info-label {{ font-size: 10px; color: var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom: 3px; }}
  .info-value {{ font-size: 14px; font-weight: 600; color: var(--text); }}

  /* ── Image Grid ── */
  .img-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-bottom: 16px;
  }}
  .img-card {{
    background: var(--card);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
  }}
  .img-card img {{ width: 100%; display: block; }}
  .img-label {{
    padding: 6px 12px;
    font-size: 11px; color: var(--muted);
    font-weight: 600; text-align: center;
    letter-spacing: 0.3px;
  }}

  /* ── Finding Card ── */
  .finding-card {{
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-left: 3px solid var(--border);
  }}
  .finding-card.ok    {{ background: var(--green-lt); border-left-color: var(--green); }}
  .finding-card.warn  {{ background: var(--amber-lt); border-left-color: var(--amber); }}
  .finding-card.alert {{ background: var(--red-lt);   border-left-color: var(--red);   }}
  .finding-title {{
    font-size: 11px; font-weight: 700; letter-spacing: 0.5px;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 6px;
  }}
  .finding-grade {{
    display: inline-block;
    padding: 2px 10px; border-radius: 4px;
    font-size: 11px; font-weight: 700;
    margin-bottom: 6px; letter-spacing: 0.3px;
  }}
  .grade-ok    {{ background: #BBF7D0; color: var(--green); }}
  .grade-warn  {{ background: #FDE68A; color: var(--amber); }}
  .grade-alert {{ background: #FCA5A5; color: var(--red);   }}
  .angle-badge {{
    display: inline-block;
    background: #1a2a3a;
    border: 1px solid var(--accent2);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 13px;
    font-weight: 700;
    color: var(--accent2);
    margin-left: 8px;
  }}

  /* ── Coaching ── */
  .coaching-list {{ display: flex; flex-direction: column; gap: 10px; }}
  .coaching-item {{
    background: linear-gradient(135deg, #0d2016, #0d1a28);
    border: 1px solid #1e4030;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
  }}

  /* ── Angle Table ── */
  .angle-table {{ width: 100%; border-collapse: collapse; }}
  .angle-table th, .angle-table td {{
    padding: 10px 14px; text-align: left;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
  }}
  .angle-table th {{ color: var(--muted); font-weight: 600; font-size: 12px; }}
  .angle-table tr:last-child td {{ border-bottom: none; }}

  /* ── Disclaimer ── */
  .disclaimer {{
    background: #16131a;
    border: 1px solid #2a2030;
    border-radius: 8px;
    padding: 16px 20px;
    color: var(--muted);
    font-size: 12px;
    margin-top: 24px;
    line-height: 1.8;
  }}

  /* ── Limit Note ── */
  .limit-note {{
    background: #1a1610;
    border: 1px solid #3a3010;
    border-radius: 6px;
    padding: 8px 12px;
    color: #a8966a;
    font-size: 12px;
    margin-top: 10px;
  }}

  /* ── Print ── */
  @media print {{
    body {{ background: #fff; color: #111; }}
    .report-header, .score-card, .section {{ border-color: #ddd; background: #fff; }}
    .section-title {{ color: #1a56a0; }}
    :root {{
      --bg: #fff; --surface: #fff; --card: #f8f8f8;
      --text: #111; --muted: #555; --border: #ddd;
    }}
  }}
</style>
</head>
<body>
<div class="report-wrap">

  <!-- Header -->
  <div class="report-header">
    <div style="display:flex;align-items:center;gap:16px;">
      {logo_html}
      <div>
        <div class="brand"><span>DailyFit</span>健身空間</div>
        <div style="color:var(--muted);font-size:12px;margin-top:2px;">體態評估分析報告</div>
      </div>
    </div>
    <div class="report-meta">
      <div class="client-name">{client_info.get('name','')}</div>
      <div>評估日期：{client_info.get('date', ts)}</div>
      <div>報告生成：{ts}</div>
    </div>
  </div>

  <!-- Score Summary -->
  <div class="score-card">
    <div class="score-circle">
      <div class="score-num">{score}</div>
      <div class="score-unit">/ 100</div>
    </div>
    <div class="score-detail">
      <div class="score-label">{label}</div>
      <div style="color:var(--muted);font-size:13px;">{score_data.get('category_description','')}</div>
      {_render_top3(top3)}
      <div class="score-note">※ {score_data.get('note','')}</div>
    </div>
  </div>

  <!-- Client Profile -->
  <div class="section">
    <div class="section-title">👤 客戶基本資料</div>
    <div class="info-grid">
      <div class="info-item"><div class="info-label">姓名</div><div class="info-value">{client_info.get('name','—')}</div></div>
      <div class="info-item"><div class="info-label">性別</div><div class="info-value">{client_info.get('gender','—')}</div></div>
      <div class="info-item"><div class="info-label">年齡</div><div class="info-value">{client_info.get('age','—')} 歲</div></div>
      <div class="info-item"><div class="info-label">身高</div><div class="info-value">{client_info.get('height','—')} cm</div></div>
      <div class="info-item"><div class="info-label">體重</div><div class="info-value">{client_info.get('weight','—')} kg</div></div>
      <div class="info-item"><div class="info-label">評估日期</div><div class="info-value">{client_info.get('date','—')}</div></div>
    </div>
    {"<div style='margin-top:12px;color:var(--muted);font-size:13px;'>備註：" + client_info.get('notes','') + "</div>" if client_info.get('notes') else ''}
  </div>

  <!-- Annotated Images Overview -->
  <div class="section">
    <div class="section-title">📸 姿勢標注圖像</div>
    <div class="img-grid">
      {_img_card(front_img, "正面站姿")}
      {_img_card(back_img,  "背面站姿")}
      {_img_card(left_img,  "左側站姿")}
      {_img_card(right_img, "右側站姿")}
    </div>
    {_img_card_single("", "")}
  </div>

  <!-- Angle Metrics Table -->
  <div class="section">
    <div class="section-title">📐 角度測量指標</div>
    <table class="angle-table">
      <tr>
        <th>測量項目</th><th>左側</th><th>右側</th><th>參考範圍</th>
      </tr>
      {_angle_row("頸椎傾斜角", left_findings, right_findings, "cervical_angle", "< 10° 接近中立")}
      {_angle_row("前傾肩角 FSA", left_findings, right_findings, "forward_shoulder_angle", "< 52° 正常")}
      {_angle_row("骨盆傾斜角（估算）", left_findings, right_findings, "pelvic_tilt_angle", "5–10° 中立")}
      {_angle_row("膝關節角度", left_findings, right_findings, "knee_angle", "> 170° 正常")}
    </table>
    <div class="limit-note">⚠️ {STATIC_ANALYSIS_LIMIT}</div>
  </div>

  <!-- Front View Findings -->
  <div class="section">
    <div class="section-title">正面姿勢分析</div>
    {_render_finding("肩部高低對稱性", front_findings.get("shoulder_asymmetry", {}))}
    {_render_finding("骨盆高低對稱性", front_findings.get("pelvic_asymmetry", {}))}
    {_render_balance_note(front_findings.get("balance_note", ""))}
    {_confidence_note(front_findings)}
  </div>

  <!-- Back View Findings -->
  <div class="section">
    <div class="section-title">背面姿勢分析</div>
    {_render_finding("肩部高低（確認）", back_findings.get("shoulder_asymmetry", {}))}
    {_render_finding("骨盆高低（確認）", back_findings.get("pelvic_asymmetry", {}))}
    <div class="limit-note">ℹ️ {back_findings.get('limit_note', '')}</div>
    {_confidence_note(back_findings)}
  </div>

  <!-- Side View Findings -->
  <div class="section">
    <div class="section-title">側面姿勢分析</div>
    <div style="color:var(--muted);font-size:12px;margin-bottom:14px;">左側與右側結合觀察，數值以可信度較高者為主。</div>
    {_render_side_findings(left_findings, right_findings)}
    {_confidence_note(left_findings)}
  </div>

  <!-- Forward Bend Test -->

  <!-- Coaching Direction -->
  <div class="section">
    <div class="section-title">🎯 訓練建議</div>
    <div class="coaching-list">
      {"".join(f'<div class="coaching-item">{s}</div>' for s in coaching)}
    </div>
  </div>

  <!-- Disclaimer -->
  <div class="disclaimer">
    <strong>系統說明與限制</strong><br><br>
    {DISCLAIMER}<br><br>
    本分析基於靜態 2D 照片，僅能提供姿勢傾向參考，無法取代動態功能性評估或臨床檢查。
    所有角度數值均為估算，並非精確量測結果。建議搭配教練動作觀察，綜合判斷。
  </div>

</div>
</body>
</html>"""

    return html


# ── 輔助 HTML 片段生成 ──────────────────────────────────

def _score_color(score: int) -> str:
    if score >= 85: return "#7FA85C"   # 草綠
    if score >= 70: return "#C87941"   # 琥珀
    if score >= 50: return "#C45A3A"   # 磚紅
    return "#A03020"                   # 深棕紅


def _render_top3(top3: list) -> str:
    if not top3:
        return "<div style='color:var(--muted);font-size:13px;margin-top:8px;'>未發現明顯姿勢偏差。</div>"
    items = ""
    for i, issue in enumerate(top3, 1):
        items += f"""
        <div class="top3-item">
          <div class="issue-source">觀察 {i}｜{issue.get('source','')}</div>
          {issue.get('description','')[:80]}{"..." if len(issue.get('description','')) > 80 else ""}
        </div>"""
    return f'<div class="top3-list">{items}</div>'


def _img_card(b64: str, label: str) -> str:
    if not b64:
        return f'<div class="img-card"><div style="height:200px;display:flex;align-items:center;justify-content:center;color:var(--muted);">{label}（未提供）</div></div>'
    return f'''
    <div class="img-card">
      <img src="{b64}" alt="{label}" loading="lazy">
      <div class="img-label">{label}</div>
    </div>'''


def _img_card_single(b64: str, label: str) -> str:
    if not b64:
        return ""
    return f'''
    <div class="img-card" style="max-width:420px;margin:0 auto;">
      <img src="{b64}" alt="{label}" loading="lazy">
      <div class="img-label">{label}</div>
    </div>'''


def _render_finding(title: str, data: dict) -> str:
    if not data or "error" in data:
        return f'<div class="finding-card"><div class="finding-title">{title}</div><div style="color:var(--muted);">無法取得分析結果。</div></div>'
    
    grade = data.get("grade", "")
    desc  = data.get("description", "")
    angle = data.get("angle")

    card_class = "ok" if grade in ("正常範圍","接近中立","接近正常","對稱") else \
                 "warn" if grade in ("輕度","輕度前傾","輕度不對稱","正常或輕微") else "alert"
    grade_class = "grade-ok" if card_class == "ok" else \
                  "grade-warn" if card_class == "warn" else "grade-alert"

    angle_html = f'<span class="angle-badge">{angle:.1f}°</span>' if angle is not None else ""

    limit = data.get("limit_note","")
    limit_html = f'<div class="limit-note" style="margin-top:8px;">⚠️ {limit}</div>' if limit else ""

    return f'''
    <div class="finding-card {card_class}">
      <div class="finding-title">{title}{angle_html}</div>
      <span class="finding-grade {grade_class}">{grade}</span>
      <div style="font-size:13px;">{desc}</div>
      {limit_html}
    </div>'''


def _render_balance_note(note: str) -> str:
    if not note:
        return ""
    return f'<div style="margin-top:12px;padding:10px 14px;background:var(--card);border-radius:8px;font-size:13px;color:var(--muted);">💬 {note}</div>'


def _render_side_findings(left: dict, right: dict) -> str:
    """合併左右側面結果顯示"""
    html = ""
    checks = [
        ("頭部前引傾向", "forward_head"),
        ("頸椎傾斜角", "cervical_angle"),
        ("前傾肩角（FSA 圓肩）", "forward_shoulder_angle"),
        ("骨盆傾斜角（估算）", "pelvic_tilt_angle"),
        ("膝關節角度", "knee_angle"),
    ]
    for title, key in checks:
        # 優先取較嚴重一側
        l_data = left.get(key, {}) if isinstance(left, dict) else {}
        r_data = right.get(key, {}) if isinstance(right, dict) else {}
        l_ded  = l_data.get("deduction", 0) if isinstance(l_data, dict) else 0
        r_ded  = r_data.get("deduction", 0) if isinstance(r_data, dict) else 0
        data   = l_data if l_ded >= r_ded else r_data
        html  += _render_finding(title, data)
    return html


def _render_bend_section(bend: dict) -> str:
    if not bend:
        return ""
    sh_sym    = bend.get("shoulder_symmetry", {})
    sp_shift  = bend.get("spine_shift", {})
    overall   = bend.get("overall", "")
    return f'''
  <div class="section">
    <div class="section-title">🏃 背向前彎動作評估</div>
    {_render_finding("肩部對稱性（前彎）", sh_sym)}
    {_render_finding("脊柱中線偏移", sp_shift)}
    {_render_balance_note(overall)}
    <div class="limit-note" style="margin-top:10px;">
      ⚠️ 本測試僅作為初步姿勢參考，疑似脊柱偏移傾向需由合格醫療人員進一步確認，本工具不構成診斷依據。
    </div>
  </div>'''


def _angle_row(label: str, left: dict, right: dict, key: str, ref: str) -> str:
    l_val = left.get(key, {}).get("angle") if isinstance(left.get(key), dict) else None
    r_val = right.get(key, {}).get("angle") if isinstance(right.get(key), dict) else None
    l_str = f"{l_val:.1f}°" if l_val is not None else "—"
    r_str = f"{r_val:.1f}°" if r_val is not None else "—"
    return f"<tr><td>{label}</td><td>{l_str}</td><td>{r_str}</td><td style='color:var(--muted)'>{ref}</td></tr>"


def _confidence_note(findings: dict) -> str:
    note = findings.get("confidence_note", "") if isinstance(findings, dict) else ""
    if note:
        return f'<div class="limit-note" style="margin-top:10px;">ℹ️ {note}</div>'
    return ""
