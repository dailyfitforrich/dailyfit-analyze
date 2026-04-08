# app.py  ─  DailyFit健身空間  v4.0
# 設計方向：現代科技感淺色介面 / 高對比文字 / 無 emoji icon

import streamlit as st
import json, traceback, os, base64
from datetime import date
from PIL import Image

from analyzer.pose_detector import PoseDetector
from analyzer.posture_rules import (
    analyze_front_view, analyze_back_view,
    analyze_side_view, aggregate_findings,
)
from analyzer.annotator import (
    annotate_front_view as ann_front,
    annotate_back_view  as ann_back,
    annotate_side_view  as ann_side,
)
from analyzer.scoring import calculate_score, get_coaching_suggestions
from analyzer.report_generator import generate_html_report
from utils.file_utils import create_output_dirs, save_assessment_json, save_html_report
import cv2

# ── HEIC 支援 ──────────────────────────────────────────────
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIC_OK = True
except ImportError:
    _HEIC_OK = False

_UPLOAD_TYPES = ["jpg","jpeg","png","heic","heif"] if _HEIC_OK else ["jpg","jpeg","png"]

# ── Logo ──────────────────────────────────────────────────
def _logo_b64(fname="logo.png") -> str:
    p = os.path.normpath(os.path.join(os.path.dirname(__file__), "assets", fname))
    if not os.path.exists(p): return ""
    mime = "jpeg" if fname.endswith(".jpg") else "png"
    with open(p,"rb") as f:
        return f"data:image/{mime};base64,"+base64.b64encode(f.read()).decode()

_LOGO = _logo_b64("logo.png")


# ─────────────────────────────────────────────────────────
# 色盤：現代科技感淺色系
#   主色  #0F52BA  深藍（科技主色）
#   強調  #00B4D8  天藍（數據亮點）
#   成功  #22C55E  綠色
#   警告  #F59E0B  琥珀
#   危險  #EF4444  紅色
#   背景  #F8FAFC  冷白
#   表面  #FFFFFF
#   邊框  #E2E8F0
#   文字  #0F172A  近黑（高對比）
#   次要  #475569  石板灰
# ─────────────────────────────────────────────────────────
C = {
    "bg":        "#F8FAFC",
    "surface":   "#FFFFFF",
    "card":      "#F1F5F9",
    "border":    "#E2E8F0",
    "blue":      "#0F52BA",
    "blue_lt":   "#EFF6FF",
    "cyan":      "#0EA5E9",
    "cyan_lt":   "#E0F2FE",
    "text":      "#0F172A",
    "sub":       "#475569",
    "muted":     "#94A3B8",
    "green":     "#16A34A",
    "green_lt":  "#DCFCE7",
    "amber":     "#D97706",
    "amber_lt":  "#FEF3C7",
    "red":       "#DC2626",
    "red_lt":    "#FEE2E2",
    "score_ok":  "#16A34A",
    "score_w":   "#D97706",
    "score_d":   "#DC2626",
    "score_e":   "#991B1B",
}


# ═══════════════════════════════════════════════════════════
# 頁面設定（第一行）
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DailyFit健身空間 · 體態分析",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ═══════════════════════════════════════════════════════════
# Helper 函式（定義在 UI 之前）
# ═══════════════════════════════════════════════════════════
_LABEL_MAP = {
    "shoulder_asymmetry":     "肩部高低對稱性",
    "pelvic_asymmetry":       "骨盆高低對稱性",
    "balance_note":           None,
    "overall":                None,
    "cervical_angle":         "頸椎傾斜角",
    "forward_shoulder_angle": "前傾肩角（FSA）",
    "pelvic_tilt_angle":      "骨盆傾斜角（估算）",
    "knee_angle":             "膝關節角度",
    "forward_head":           "頭部前引傾向",
}

_OK_GRADES = {"正常範圍","接近中立","接近正常","對稱","未觀察到明顯偏移","正常或輕微"}


def _score_hex(s: int) -> str:
    if s >= 85: return C["score_ok"]
    if s >= 70: return C["score_w"]
    if s >= 50: return C["score_d"]
    return C["score_e"]


def _grade_style(grade: str):
    """回傳 (card_bg, left_border, badge_bg, badge_txt)"""
    if grade in _OK_GRADES:
        return C["green_lt"], C["green"], "#D1FAE5", C["green"]
    if any(w in grade for w in ["明顯","疑似","無法"]):
        return C["red_lt"],   C["red"],   "#FEE2E2", C["red"]
    return C["amber_lt"],  C["amber"],  "#FEF3C7", C["amber"]


def _finding_card(label: str, data: dict):
    grade = data.get("grade","")
    desc  = data.get("description","")
    angle = data.get("angle")
    limit = data.get("limit_note","")
    bg, border, bbg, btxt = _grade_style(grade)

    angle_html = (
        f' <span style="font-weight:700;color:{C["blue"]};font-size:15px;">'
        f'{angle:.1f}°</span>'
    ) if angle is not None else ""

    limit_html = (
        f'<div style="font-size:11px;color:{C["amber"]};margin-top:6px;'
        f'padding:4px 8px;background:{C["amber_lt"]};border-radius:4px;">'
        f'注意：{limit}</div>'
    ) if limit else ""

    st.markdown(f"""
    <div style="border-left:4px solid {border};background:{bg};
                border-radius:0 8px 8px 0;padding:14px 18px;margin-bottom:10px;">
      <div style="font-size:12px;font-weight:600;color:{C['sub']};
                  text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">
        {label}{angle_html}
      </div>
      <span style="display:inline-block;background:{bbg};color:{btxt};
                   border-radius:4px;padding:2px 10px;font-size:11px;
                   font-weight:700;letter-spacing:0.3px;margin-bottom:8px;">{grade}</span>
      <div style="font-size:13px;color:{C['text']};line-height:1.7;">{desc}</div>
      {limit_html}
    </div>
    """, unsafe_allow_html=True)


def _render_findings_tab(findings: dict, title: str = ""):
    if title:
        st.markdown(f"#### {title}")
    if not findings:
        st.info("無分析結果。"); return
    if "error" in findings:
        st.error(findings["error"]); return

    for key, data in findings.items():
        if key in ("confidence_note","limit_note","error","side"): continue
        display = _LABEL_MAP.get(key, key)
        if display is None:
            if isinstance(data, str) and data:
                st.markdown(
                    f'<div style="color:{C["sub"]};font-size:13px;'
                    f'padding:8px 12px;background:{C["card"]};'
                    f'border-radius:6px;margin-bottom:8px;">{data}</div>',
                    unsafe_allow_html=True)
            continue
        if not isinstance(data, dict): continue
        _finding_card(display, data)

    note = findings.get("confidence_note","")
    if note:
        st.markdown(
            f'<div style="font-size:12px;color:{C["sub"]};padding:8px 12px;'
            f'background:{C["cyan_lt"]};border-left:3px solid {C["cyan"]};'
            f'border-radius:0 6px 6px 0;margin-top:8px;">{note}</div>',
            unsafe_allow_html=True)


def _render_angle_table(left_f, right_f):
    import pandas as pd
    rows = [
        ("頸椎傾斜角",        "cervical_angle",         "< 10° 中立 / 10–20° 輕度 / > 20° 明顯"),
        ("前傾肩角 FSA",       "forward_shoulder_angle", "< 52° 正常 / ≥ 52° 圓肩傾向"),
        ("骨盆傾斜角（估算）", "pelvic_tilt_angle",      "0–8° 中立"),
        ("膝關節角度",         "knee_angle",              "> 170° 正常"),
    ]
    table = []
    for label, key, ref in rows:
        lv = (left_f.get(key) or {}).get("angle")
        rv = (right_f.get(key) or {}).get("angle")
        table.append({"測量項目":label,
                      "左側":f"{lv:.1f}°" if lv else "—",
                      "右側":f"{rv:.1f}°" if rv else "—",
                      "參考範圍":ref})
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    st.markdown(
        f'<p style="font-size:11px;color:{C["muted"]};margin-top:4px;">'
        "所有數值為靜態 2D 估算，具固有限制，僅供教練參考。</p>",
        unsafe_allow_html=True)


def _cv2_pil(arr): return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def _load_image(src) -> Image.Image:
    """
    接受多種來源並回傳 PIL RGB Image：
    - PIL Image（camera_input 解析後 / session_state 快取）
    - UploadedFile（file_uploader 回傳）
    """
    if isinstance(src, Image.Image):
        return src.convert("RGB")
    # UploadedFile or BytesIO
    name = getattr(src, "name", "").lower()
    if name.endswith((".heic", ".heif")):
        if _HEIC_OK:
            return Image.open(src).convert("RGB")
        st.error("HEIC 格式需安裝 pillow-heif 套件"); st.stop()
    return Image.open(src).convert("RGB")


# ═══════════════════════════════════════════════════════════
# 全域 CSS
# ═══════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+TC:wght@400;500;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Noto Sans TC', 'Inter', -apple-system, sans-serif;
  }}

  /* 整體背景：冷白，像 SaaS 產品 */
  .main {{ background: {C['bg']} !important; }}
  .block-container {{
    padding-top: 1.2rem;
    padding-bottom: 3rem;
    max-width: 1080px;
  }}

  /* ── Header ── */
  .df-header {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 18px 28px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .df-logo {{ height: 52px; width: auto; flex-shrink: 0; }}
  .df-brand {{
    font-size: 20px; font-weight: 700;
    color: {C['text']}; letter-spacing: -0.3px;
  }}
  .df-brand .accent {{ color: {C['blue']}; }}
  .df-sub {{
    font-size: 12px; color: {C['muted']};
    margin-top: 2px; letter-spacing: 0.2px;
  }}
  .df-tagline {{
    margin-left: auto;
    font-size: 12px; color: {C['sub']};
    text-align: right; line-height: 1.8;
  }}

  /* ── 區段標題 ── */
  .sec-label {{
    font-size: 11px; font-weight: 700;
    color: {C['blue']}; letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 12px; margin-top: 4px;
  }}

  /* ── 必填 / 選填標籤 ── */
  .tag-req {{
    background: {C['blue_lt']}; color: {C['blue']};
    font-size: 10px; font-weight: 600;
    padding: 1px 7px; border-radius: 4px; margin-left: 6px;
    letter-spacing: 0.3px;
  }}
  .tag-opt {{
    background: {C['green_lt']}; color: {C['green']};
    font-size: 10px; font-weight: 600;
    padding: 1px 7px; border-radius: 4px; margin-left: 6px;
  }}

  /* ── 開始分析按鈕 ── */
  div.stButton > button {{
    background: {C['blue']};
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 12px 28px;
    font-size: 15px; font-weight: 600;
    width: 100%;
    letter-spacing: 0.3px;
    transition: all 0.15s;
    box-shadow: 0 2px 8px rgba(15,82,186,0.25);
  }}
  div.stButton > button:hover {{
    background: #0A3D99;
    box-shadow: 0 4px 16px rgba(15,82,186,0.35);
    transform: translateY(-1px);
  }}
  div.stButton > button:disabled {{
    background: {C['muted']}; box-shadow: none; transform: none;
  }}

  /* ── 輸入欄位 ── */
  [data-testid="stTextInput"] input {{
    background: {C['surface']};
    color: {C['text']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    font-size: 14px;
  }}
  [data-testid="stTextInput"] input:focus {{
    border-color: {C['blue']};
    box-shadow: 0 0 0 3px {C['blue_lt']};
  }}

  /* ── Divider ── */
  hr {{ border-color: {C['border']} !important; margin: 20px 0 !important; }}

  /* ── Tabs ── */
  [data-testid="stTabs"] [data-baseweb="tab"] {{
    font-size: 13px; font-weight: 600; color: {C['sub']};
  }}
  [data-testid="stTabs"] [aria-selected="true"] {{
    color: {C['blue']};
    border-bottom-color: {C['blue']};
  }}

  /* ── Metric cards ── */
  .metric-card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  }}

  /* ── Coaching item ── */
  .coaching-item {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 14px;
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 13px;
    color: {C['text']};
    line-height: 1.6;
  }}
  .coaching-dot {{
    width: 6px; height: 6px;
    background: {C['blue']};
    border-radius: 50%;
    margin-top: 7px;
    flex-shrink: 0;
  }}

  /* ── Top3 item ── */
  .top3-item {{
    padding: 10px 14px;
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-left: 3px solid {C['blue']};
    border-radius: 0 8px 8px 0;
    margin-bottom: 8px;
    font-size: 13px;
    color: {C['text']};
    line-height: 1.6;
  }}
  .top3-index {{
    font-size: 11px; font-weight: 700;
    color: {C['blue']}; letter-spacing: 0.5px;
    margin-bottom: 3px;
  }}

  /* Streamlit chrome */
  #MainMenu {{ visibility: hidden; }}
  footer    {{ visibility: hidden; }}
  header    {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════
if "done" not in st.session_state: st.session_state.done = False
if "res"  not in st.session_state: st.session_state.res  = {}


# ═══════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════
logo_html = (f'<img src="{_LOGO}" class="df-logo" alt="DailyFit">' if _LOGO else "")
st.markdown(f"""
<div class="df-header">
  {logo_html}
  <div>
    <div class="df-brand"><span class="accent">DailyFit</span>健身空間</div>
    <div class="df-sub">體態評估分析報告系統 · FITNESS POSTURE ASSESSMENT</div>
  </div>
  <div class="df-tagline">
    上傳四面站姿照片<br>自動生成體態觀察報告
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# 客戶資料表單
# ═══════════════════════════════════════════════════════════
st.markdown('<div class="sec-label">客戶基本資料</div>', unsafe_allow_html=True)

r1, r2, r3 = st.columns([2,1,1])
with r1: inp_name   = st.text_input("姓名", placeholder="請輸入客戶姓名", key="inp_name")
with r2: inp_gender = st.selectbox("性別", ["男性","女性","其他"], key="inp_gender")
with r3: inp_age    = st.number_input("年齡", 10, 90, 30, key="inp_age")

r4, r5, r6 = st.columns([1,1,2])
with r4: inp_height = st.number_input("身高（cm）",100.0,220.0,170.0,0.5,key="inp_h")
with r5: inp_weight = st.number_input("體重（kg）", 30.0,200.0, 65.0,0.5,key="inp_w")
with r6: inp_date   = st.date_input("評估日期", value=date.today(), key="inp_date")

inp_notes = st.text_input("備註（選填）",
    placeholder="例如：慣用手為右手、有腰痛病史等", key="inp_notes")
st.markdown("---")


# ═══════════════════════════════════════════════════════════
# 照片輸入（拍照 / 上傳 兩種模式，相容 iOS & Android）
# ═══════════════════════════════════════════════════════════
heic_tip = "JPG、PNG、HEIC" if _HEIC_OK else "JPG、PNG"

st.markdown('<div class="sec-label">姿勢照片</div>', unsafe_allow_html=True)
st.markdown(
    f'<p style="font-size:13px;color:{C["sub"]};margin-bottom:4px;">'
    f"每個角度可選擇「直接拍照」或「從相簿 / 裝置選取」。"
    f"全身直式拍攝，鏡頭距離建議 2–3 公尺。</p>",
    unsafe_allow_html=True)

# ── 全域模式切換（一鍵切換四個槽）──────────────────────────
mode_cols = st.columns([2, 1])
with mode_cols[1]:
    global_mode = st.radio(
        "輸入方式",
        ["拍照", "從相簿上傳"],
        horizontal=True,
        key="global_mode",
        label_visibility="collapsed",
    )
st.markdown("")

# ── 各角度定義 ───────────────────────────────────────────
VIEWS = [
    ("正面站姿", "雙腳與肩同寬，雙臂自然下垂", "front"),
    ("背面站姿", "背對鏡頭，姿勢與正面相同",   "back"),
    ("左側站姿", "左側面對鏡頭，全身直立",     "left"),
    ("右側站姿", "右側面對鏡頭，全身直立",     "right"),
]

# ── 拍照模式提示（相機 widget 佔空間較大，改為兩排顯示）──
if global_mode == "拍照":
    # 手機拍照：兩欄兩排，畫面不擠
    row1 = st.columns(2)
    row2 = st.columns(2)
    view_cols = [row1[0], row1[1], row2[0], row2[1]]
else:
    view_cols = st.columns(4)

# ── 初始化 session state ──────────────────────────────────
for _, _, key in VIEWS:
    if f"photo_{key}" not in st.session_state:
        st.session_state[f"photo_{key}"] = None   # 儲存最終 PIL Image

def _resolve_photo(key: str):
    """
    回傳該槽的最終圖片（PIL Image 或 None）。
    依序嘗試：camera_input → file_uploader → session_state 快取
    """
    cam  = st.session_state.get(f"cam_{key}")
    file = st.session_state.get(f"file_{key}")
    src  = cam if cam is not None else file
    if src is not None:
        try:
            img = Image.open(src).convert("RGB")
            st.session_state[f"photo_{key}"] = img
            return img
        except Exception:
            pass
    return st.session_state.get(f"photo_{key}")

# ── 每個角度的輸入欄 ──────────────────────────────────────
collected = {}

for (label, hint, key), col in zip(VIEWS, view_cols):
    with col:
        # 欄位標題
        st.markdown(
            f'<p style="font-size:13px;font-weight:600;color:{C["text"]};'
            f'margin-bottom:1px;margin-top:4px;">'
            f'{label} <span class="tag-req">必填</span></p>'
            f'<p style="font-size:11px;color:{C["muted"]};margin-bottom:6px;">{hint}</p>',
            unsafe_allow_html=True)

        if global_mode == "拍照":
            # ── 相機模式 ──────────────────────────────────
            # st.camera_input：
            #   iOS Safari → 開啟相機 App
            #   Android Chrome → 開啟相機
            #   桌機 → 開啟 Webcam
            cam_img = st.camera_input(
                f"拍攝{label}",
                key=f"cam_{key}",
                label_visibility="collapsed",
            )
            if cam_img:
                st.success("已拍攝", icon=None)
            elif st.session_state.get(f"photo_{key}"):
                st.markdown(
                    f'<p style="font-size:11px;color:{C["green"]};">已有照片（可重新拍攝）</p>',
                    unsafe_allow_html=True)
        else:
            # ── 上傳模式 ──────────────────────────────────
            # file_uploader：
            #   iOS → 可選「拍照」或「從相簿選取」
            #   Android → 可選相機或檔案
            #   桌機 → 開啟檔案選擇器
            file_img = st.file_uploader(
                f"上傳{label}",
                type=_UPLOAD_TYPES,
                key=f"file_{key}",
                label_visibility="collapsed",
                help=f"支援格式：{heic_tip}",
            )
            if file_img:
                try:
                    preview = Image.open(file_img)
                    st.image(preview, use_container_width=True)
                except Exception:
                    pass
            elif st.session_state.get(f"photo_{key}"):
                st.markdown(
                    f'<p style="font-size:11px;color:{C["green"]};">已有照片（可重新上傳）</p>',
                    unsafe_allow_html=True)

        # 解析最終圖片
        collected[key] = _resolve_photo(key)

front_file = collected["front"]
back_file  = collected["back"]
left_file  = collected["left"]
right_file = collected["right"]

# ── 已收集照片狀態列 ──────────────────────────────────────
status_items = []
for label, _, key in VIEWS:
    img = collected[key]
    if img:
        status_items.append(
            f'<span style="color:{C["green"]};font-weight:600;">'
            f'{label} ✓</span>')
    else:
        status_items.append(
            f'<span style="color:{C["muted"]};">{label} —</span>')

st.markdown(
    f'<div style="background:{C["card"]};border:1px solid {C["border"]};'
    f'border-radius:8px;padding:8px 16px;margin-top:12px;font-size:12px;">'
    f'照片狀態：{"　".join(status_items)}</div>',
    unsafe_allow_html=True)

st.markdown("---")


# ═══════════════════════════════════════════════════════════
# 驗證 + 按鈕
# ═══════════════════════════════════════════════════════════
required = {"正面":front_file,"背面":back_file,"左側":left_file,"右側":right_file}
missing  = [k for k,v in required.items() if v is None]

if missing:
    st.markdown(
        f'<div style="background:{C["amber_lt"]};border:1px solid {C["amber"]}44;'
        f'border-radius:8px;padding:10px 14px;font-size:13px;color:{C["amber"]};">'
        f'尚未取得：{"、".join(missing)} 照片，請拍攝或上傳後再開始分析。</div>',
        unsafe_allow_html=True)
if not inp_name:
    st.markdown(
        f'<div style="background:{C["blue_lt"]};border:1px solid {C["blue"]}33;'
        f'border-radius:8px;padding:10px 14px;font-size:13px;color:{C["blue"]};">'
        f'請先填寫客戶姓名。</div>',
        unsafe_allow_html=True)

btn_col, _ = st.columns([1, 2])
with btn_col:
    run_btn = st.button("開始分析", disabled=(bool(missing) or not inp_name))


# ═══════════════════════════════════════════════════════════
# 分析執行
# ═══════════════════════════════════════════════════════════
if run_btn:
    create_output_dirs()
    detector = PoseDetector()
    bar = st.progress(0, text="正在初始化...")
    client_info = {
        "name":inp_name, "gender":inp_gender, "age":int(inp_age),
        "height":float(inp_height), "weight":float(inp_weight),
        "date":str(inp_date), "notes":inp_notes,
    }
    try:
        imgs = {
            "front":_load_image(front_file), "back":_load_image(back_file),
            "left":_load_image(left_file),   "right":_load_image(right_file),
        }
        labels = {"front":"正面","back":"背面","left":"左側","right":"右側"}
        det = {}
        for i,(v,img) in enumerate(imgs.items()):
            bar.progress(8+i*16, text=f"偵測 {labels[v]} 姿勢地標...")
            det[v] = detector.detect(img)

        bar.progress(72, text="計算姿勢角度與觀察...")
        front_f = analyze_front_view(det["front"])
        back_f  = analyze_back_view(det["back"])
        left_f  = analyze_side_view(det["left"],  "left")
        right_f = analyze_side_view(det["right"], "right")

        aggregated = aggregate_findings(front_f, back_f, left_f, right_f, None)
        score_data = calculate_score(aggregated)

        bar.progress(84, text="生成標注圖像...")
        annotated = {
            "front": ann_front(det["front"], front_f),
            "back":  ann_back(det["back"],   back_f),
            "left":  ann_side(det["left"],   left_f,  "left"),
            "right": ann_side(det["right"],  right_f, "right"),
        }

        bar.progress(94, text="生成分析報告...")
        html_report = generate_html_report(
            client_info=client_info,
            front_findings=front_f, back_findings=back_f,
            left_findings=left_f,   right_findings=right_f,
            bend_findings=None,     aggregated=aggregated,
            score_data=score_data,  annotated_images=annotated,
            original_images=imgs,
        )
        save_html_report(html_report, inp_name)
        save_assessment_json(client_info,
            {"front":front_f,"back":back_f,"left":left_f,"right":right_f},
            score_data)

        bar.progress(100, text="分析完成")
        st.session_state.done = True
        st.session_state.res  = {
            "client_info":client_info,
            "front_f":front_f, "back_f":back_f,
            "left_f":left_f,   "right_f":right_f,
            "aggregated":aggregated, "score_data":score_data,
            "annotated":annotated,   "html_report":html_report,
        }
    except Exception as e:
        bar.empty()
        st.error(f"分析過程發生錯誤：{e}")
        with st.expander("錯誤詳情"):
            st.code(traceback.format_exc())
        st.stop()
    finally:
        detector.close()


# ═══════════════════════════════════════════════════════════
# 結果顯示
# ═══════════════════════════════════════════════════════════
if st.session_state.done and st.session_state.res:
    r          = st.session_state.res
    score_data = r["score_data"]
    aggregated = r["aggregated"]
    score      = score_data["score"]
    label      = score_data["label"]
    name       = r["client_info"]["name"]
    sc         = _score_hex(score)
    coaching   = get_coaching_suggestions(aggregated)
    top3       = aggregated.get("top3", [])

    st.markdown("---")
    st.markdown(
        f'<div class="sec-label" style="text-align:center;margin-bottom:20px;">'
        f'評估摘要</div>',
        unsafe_allow_html=True)

    # ── 三欄摘要 ───────────────────────────────────────────
    s1, s2, s3 = st.columns(3)

    # 欄 1：分數
    with s1:
        cat = score_data.get("category_description","")
        # 分數儀表盤風格
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;">
          <div style="font-size:11px;font-weight:700;color:{C['sub']};
                      letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;">
            姿勢平衡分數
          </div>
          <div style="position:relative;display:inline-block;">
            <div style="font-size:64px;font-weight:700;color:{sc};
                        line-height:1;letter-spacing:-2px;">{score}</div>
            <div style="font-size:14px;color:{C['muted']};margin-top:2px;">/ 100</div>
          </div>
          <div style="margin-top:12px;">
            <span style="display:inline-block;background:{sc}18;color:{sc};
                         border:1px solid {sc}44;border-radius:6px;
                         padding:4px 14px;font-size:13px;font-weight:700;">
              {label}
            </span>
          </div>
          <div style="font-size:12px;color:{C['sub']};margin-top:10px;
                      padding-top:10px;border-top:1px solid {C['border']};">
            {cat}
          </div>
          <div style="font-size:10px;color:{C['muted']};margin-top:6px;">
            本分數為教練內部參考指標
          </div>
        </div>
        """, unsafe_allow_html=True)

    # 欄 2：前三項觀察
    with s2:
        items_html = ""
        if not top3:
            items_html = (
                f'<div style="color:{C["green"]};font-size:13px;'
                f'padding:10px 0;">未觀察到明顯姿勢偏差</div>')
        for i, issue in enumerate(top3, 1):
            src  = issue.get("source","")
            desc = issue.get("description","")
            desc_short = desc[:60] + ("..." if len(desc)>60 else "")
            items_html += (
                f'<div class="top3-item">'
                f'<div class="top3-index">觀察 {i} · {src}</div>'
                f'{desc_short}</div>')

        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size:11px;font-weight:700;color:{C['sub']};
                      letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;">
            主要觀察
          </div>
          {items_html}
        </div>
        """, unsafe_allow_html=True)

    # 欄 3：訓練建議（高對比深色文字）
    with s3:
        items_html = "".join(
            f'<div class="coaching-item">'
            f'<div class="coaching-dot"></div>'
            f'<div style="color:{C["text"]};font-size:13px;line-height:1.6;">{t}</div>'
            f'</div>'
            for t in coaching[:4]
        )
        st.markdown(f"""
        <div class="metric-card">
          <div style="font-size:11px;font-weight:700;color:{C['sub']};
                      letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;">
            建議訓練方向
          </div>
          {items_html}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Tabs ─────────────────────────────────────────────
    tabs = st.tabs(["標注圖像", "正面分析", "背面分析", "側面分析", "下載報告"])

    with tabs[0]:
        st.markdown(
            f'<p style="font-size:13px;color:{C["sub"]};margin-bottom:14px;">'
            "關節地標與測量線已自動標注於照片上，可搭配右側分析數值說明。</p>",
            unsafe_allow_html=True)
        ic1, ic2, ic3, ic4 = st.columns(4)
        ann = r["annotated"]
        for col, key, cap in [(ic1,"front","正面"),(ic2,"back","背面"),
                              (ic3,"left","左側"),(ic4,"right","右側")]:
            with col:
                arr = ann.get(key)
                if arr is not None:
                    st.image(_cv2_pil(arr), caption=cap, use_container_width=True)
                else:
                    st.info(f"{cap}（無標注圖）")

    with tabs[1]:
        _render_findings_tab(r["front_f"], "正面姿勢分析")

    with tabs[2]:
        _render_findings_tab(r["back_f"], "背面姿勢分析")
        note = (r["back_f"] or {}).get("limit_note","")
        if note:
            st.markdown(
                f'<div style="font-size:12px;color:{C["sub"]};padding:8px 12px;'
                f'background:{C["card"]};border-radius:6px;margin-top:8px;">{note}</div>',
                unsafe_allow_html=True)

    with tabs[3]:
        lc, rc = st.columns(2)
        with lc:
            st.markdown(f'<p style="font-weight:600;color:{C["text"]};">左側</p>',
                        unsafe_allow_html=True)
            _render_findings_tab(r["left_f"])
        with rc:
            st.markdown(f'<p style="font-weight:600;color:{C["text"]};">右側</p>',
                        unsafe_allow_html=True)
            _render_findings_tab(r["right_f"])
        st.markdown("---")
        st.markdown(f'<div class="sec-label">角度測量對照</div>', unsafe_allow_html=True)
        _render_angle_table(r["left_f"], r["right_f"])

    with tabs[4]:
        st.markdown(
            f'<div class="sec-label">下載分析報告</div>', unsafe_allow_html=True)

        st.markdown(
            f'<p style="font-size:13px;color:{C["sub"]};margin-bottom:16px;">'
            "HTML 報告可在瀏覽器開啟，並透過瀏覽器列印功能轉存為 PDF。</p>",
            unsafe_allow_html=True)

        st.download_button(
            label="下載 HTML 報告",
            data=r["html_report"].encode("utf-8"),
            file_name=f"DailyFit_{name}_{date.today()}.html",
            mime="text/html",
            use_container_width=True,
            key="dl_html")

        json_export = {
            "client":r["client_info"], "score":r["score_data"],
            "top3":top3, "breakdown":score_data.get("breakdown",[])}
        st.download_button(
            label="下載 JSON 評估資料",
            data=json.dumps(json_export, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"DailyFit_{name}_{date.today()}.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json")

        st.markdown("---")
        st.markdown(
            f'<div style="background:{C["card"]};border:1px solid {C["border"]};'
            f'border-radius:8px;padding:16px 20px;'
            f'font-size:12px;color:{C["sub"]};line-height:1.9;">'
            "<strong>系統說明與限制</strong><br><br>"
            "本報告僅供健身指導參考使用，不構成醫療診斷或治療建議。"
            "所有分析結果均為姿勢傾向觀察，基於靜態 2D 照片分析，具有固有限制。"
            "建議搭配教練動作觀察與功能性評估，綜合判斷客戶狀況。"
            "</div>",
            unsafe_allow_html=True)
