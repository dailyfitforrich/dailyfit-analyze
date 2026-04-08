# utils/constants.py
# ============================================================
# DailyFit Posture Report MVP — 設定常數與閾值
# 所有分析閾值集中於此，方便後續調整
# ============================================================

# ── 肩部不對稱閾值（以肩寬比例正規化）──
SHOULDER_ASYMMETRY_MILD       = 0.03   # 輕度
SHOULDER_ASYMMETRY_MODERATE   = 0.06   # 中度
SHOULDER_ASYMMETRY_SIGNIFICANT = 0.10  # 明顯

# ── 骨盆不對稱閾值（以髖寬比例正規化）──
PELVIC_ASYMMETRY_MILD         = 0.03
PELVIC_ASYMMETRY_MODERATE     = 0.06
PELVIC_ASYMMETRY_SIGNIFICANT  = 0.10

# ── 頸椎前傾角閾值（度）──
CERVICAL_ANGLE_NORMAL = 10   # < 10° 接近中立
CERVICAL_ANGLE_MILD   = 20   # 10–20° 輕度
# > 20° 明顯頭部前引

# ── 前傾肩角 FSA 閾值（度）──
FSA_THRESHOLD = 52   # ≥ 52° 顯示明顯圓肩傾向

# ── 骨盆傾斜角閾值（度）──
PELVIC_TILT_NEUTRAL_MIN = 5
PELVIC_TILT_NEUTRAL_MAX = 10
PELVIC_TILT_ANTERIOR    = 20   # > 20° 骨盆前傾傾向
# ≤ 0° 骨盆後傾傾向

# ── 頭部前引（ear-to-shoulder 水平偏移）──
FORWARD_HEAD_MILD_RATIO     = 0.04
FORWARD_HEAD_MODERATE_RATIO = 0.08

# ── 膝關節過伸閾值（膝關節角度，超過 180° 判定為過伸傾向）──
KNEE_HYPEREXTENSION_THRESHOLD = 175  # 低於此值判定為可能過伸

# ── 脊柱側彎參考（前彎測試肩部偏差）──
SCOLIOSIS_MILD_RATIO      = 0.04
SCOLIOSIS_MODERATE_RATIO  = 0.08

# ── 計分扣分規則 ──
SCORE_DEDUCTION_MILD        = 5
SCORE_DEDUCTION_MODERATE    = 10
SCORE_DEDUCTION_SIGNIFICANT = 15

# ── 計分類別 ──
SCORE_EXCELLENT = 85   # 85–100 姿勢相對平衡
SCORE_GOOD      = 70   # 70–84  輕度不平衡傾向
SCORE_FAIR      = 50   # 50–69  明顯不平衡傾向
# < 50  多項可見不平衡傾向

# ── 輸出路徑 ──
OUTPUT_ANNOTATED      = "outputs/annotated"
OUTPUT_REPORTS_HTML   = "outputs/reports/html"
OUTPUT_REPORTS_PDF    = "outputs/reports/pdf"
OUTPUT_DATA           = "outputs/data"

# ── 圖像處理 ──
ANNOTATION_WIDTH  = 640   # 輸出標注圖像寬度
ANNOTATION_HEIGHT = 960   # 輸出標注圖像高度

# ── MediaPipe 偵測最低信心閾值 ──
MIN_DETECTION_CONFIDENCE  = 0.5
MIN_TRACKING_CONFIDENCE   = 0.5
LANDMARK_VISIBILITY_THRESHOLD = 0.5  # 低於此值視為不可靠地標

# ── 標注顏色（BGR for OpenCV）──
COLOR_LANDMARK      = (0, 255, 136)   # 螢光綠
COLOR_SHOULDER_LINE = (0, 215, 255)   # 金黃
COLOR_HIP_LINE      = (80, 107, 255)  # 珊瑚紅
COLOR_MEASURE_LINE  = (255, 210, 70)  # 青藍
COLOR_SPINE_LINE    = (200, 100, 255) # 紫
COLOR_TEXT_BG       = (30, 30, 30)    # 深灰背景
COLOR_WHITE         = (255, 255, 255)
COLOR_ANGLE_ARC     = (0, 165, 255)   # 橙色
