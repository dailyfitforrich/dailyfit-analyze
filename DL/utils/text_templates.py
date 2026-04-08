# utils/text_templates.py
# ============================================================
# DailyFit Posture Report MVP — 繁體中文文案範本
# ============================================================

# ── App 標題與說明 ──
APP_TITLE        = "DailyFit 體態分析報告系統"
APP_SUBTITLE     = "專為健身教練設計的體驗課體態評估工具"
APP_DESCRIPTION  = "上傳四面站姿照片，快速生成體態觀察報告，協助教練進行專業說明。"
DISCLAIMER       = (
    "本報告僅供健身指導參考使用，不構成醫療診斷或治療建議。"
    "所有分析結果均為姿勢傾向觀察，並非臨床評估。"
    "如有需要，請諮詢合格醫療專業人員。"
)

# ── 表單標籤 ──
LABEL_CLIENT_NAME   = "姓名"
LABEL_GENDER        = "性別"
LABEL_AGE           = "年齡"
LABEL_HEIGHT        = "身高（cm）"
LABEL_WEIGHT        = "體重（kg）"
LABEL_ASSESSMENT_DATE = "評估日期"
LABEL_NOTES         = "備註（選填）"

GENDER_OPTIONS = ["男性", "女性", "其他"]

# ── 圖片上傳標籤 ──
LABEL_FRONT_VIEW  = "正面站姿"
LABEL_BACK_VIEW   = "背面站姿"
LABEL_LEFT_VIEW   = "左側站姿"
LABEL_RIGHT_VIEW  = "右側站姿"
LABEL_BEND_VIEW   = "背向前彎（選填）"

HINT_FRONT   = "請直立站好，雙腳與肩同寬，雙臂自然下垂。"
HINT_BACK    = "背對鏡頭，姿勢與正面相同。"
HINT_LEFT    = "左側面對鏡頭，全身直立。"
HINT_RIGHT   = "右側面對鏡頭，全身直立。"
HINT_BEND    = "背對鏡頭，雙腳與肩同寬，膝蓋自然伸直，向前彎曲身體，雙手自然下垂。"

# ── 按鈕與狀態標籤 ──
BTN_RUN_ANALYSIS   = "🔍 開始分析"
BTN_DOWNLOAD_HTML  = "📄 下載 HTML 報告"
BTN_DOWNLOAD_JSON  = "💾 下載 JSON 資料"
STATUS_ANALYZING   = "正在進行體態分析，請稍候..."
STATUS_DONE        = "✅ 分析完成！"
STATUS_MISSING_IMG = "⚠️ 請先上傳所有必要照片（正面、背面、左側、右側）"

# ── 報告區段標題 ──
SECTION_SUMMARY        = "📊 評估摘要"
SECTION_CLIENT_PROFILE = "👤 客戶基本資料"
SECTION_FRONT_FINDINGS = "正面姿勢分析"
SECTION_BACK_FINDINGS  = "背面姿勢分析"
SECTION_SIDE_FINDINGS  = "側面姿勢分析"
SECTION_ANGLE_METRICS  = "角度測量指標"
SECTION_BEND_FINDINGS  = "背向前彎動作評估"
SECTION_COACHING       = "🎯 訓練建議"
SECTION_DISCLAIMER     = "系統說明與限制"

# ── 分數類別文字 ──
def get_score_label(score: int) -> tuple[str, str]:
    """回傳 (類別文字, 說明)"""
    if score >= 85:
        return "姿勢相對平衡", "整體左右對稱性良好，無明顯代償傾向。"
    elif score >= 70:
        return "輕度不平衡傾向", "存在少數輕度姿勢偏差，建議針對性訓練改善。"
    elif score >= 50:
        return "明顯不平衡傾向", "多處姿勢觀察需關注，建議優先排入矯正訓練。"
    else:
        return "多項可見不平衡傾向", "多個面向均顯示偏差傾向，建議系統性評估與訓練規劃。"

# ── 訓練建議範本（移除 emoji icon）──
COACHING_SUGGESTIONS = {
    "forward_head":         "加強呼吸控制與頸部深層肌群啟動，改善頭部前引習慣。",
    "rounded_shoulder":     "強化上背部肌群（菱形肌、中下斜方肌），改善圓肩姿勢與胸椎活動度。",
    "anterior_pelvic_tilt": "加強核心深層穩定（腹橫肌、骨盆底），改善骨盆前傾傾向與腰椎過度前凸。",
    "shoulder_asymmetry":   "針對肩部左右不對稱，建議單側訓練與肩帶穩定動作，改善習慣性偏側使用。",
    "pelvic_asymmetry":     "骨盆高低不對稱觀察，建議評估髖部活動度與單腳穩定訓練。",
    "knee_hyperextension":  "加強股四頭肌與膕旁肌協同收縮訓練，改善膝關節過伸代償。",
    "scoliosis_tendency":   "訓練上先以脊柱穩定為主，如有需要建議諮詢醫療專業人員評估。",
}

COACHING_GENERAL = [
    "建議訓練初期以動作控制與身體感知為主，避免過早加重。",
    "規律評估體態變化，追蹤訓練成效。",
]

# ── 信心不足備注 ──
LOW_CONFIDENCE_NOTE = "（圖像地標信心偏低，本項結果僅供參考）"
BACK_VIEW_LIMIT_NOTE = "背面偵測地標信心有限，背面分析結果僅供參考，建議以正面與側面結果為主。"
STATIC_ANALYSIS_LIMIT = "本分析基於靜態 2D 照片，具有固有限制，所有數值均為估算參考，非精確量測。"
