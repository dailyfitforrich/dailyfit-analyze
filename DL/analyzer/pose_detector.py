# analyzer/pose_detector.py
# ============================================================
# DailyFit Posture Report MVP — MediaPipe 姿勢偵測器
# ============================================================

import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
from typing import Optional

from utils.constants import (
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, LANDMARK_VISIBILITY_THRESHOLD
)
from utils.image_utils import pil_to_cv2


# MediaPipe Pose 地標索引
class LM:
    """MediaPipe Pose 地標索引常數"""
    NOSE          = 0
    LEFT_EYE      = 2
    RIGHT_EYE     = 5
    LEFT_EAR      = 7
    RIGHT_EAR     = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER= 12
    LEFT_ELBOW    = 13
    RIGHT_ELBOW   = 14
    LEFT_WRIST    = 15
    RIGHT_WRIST   = 16
    LEFT_HIP      = 23
    RIGHT_HIP     = 24
    LEFT_KNEE     = 25
    RIGHT_KNEE    = 26
    LEFT_ANKLE    = 27
    RIGHT_ANKLE   = 28
    LEFT_HEEL     = 29
    RIGHT_HEEL    = 30
    LEFT_FOOT     = 31
    RIGHT_FOOT    = 32


class PoseDetector:
    """
    封裝 MediaPipe Pose 靜態圖像偵測。
    
    使用方式：
        detector = PoseDetector()
        result = detector.detect(pil_image)
        if result.landmarks:
            ...
    """

    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,          # 最高精度模型
            enable_segmentation=False,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, pil_image: Image.Image) -> "DetectionResult":
        """
        對 PIL Image 執行姿勢偵測。
        回傳 DetectionResult，包含 landmarks 與 image_shape。
        """
        cv2_img = pil_to_cv2(pil_image)
        h, w = cv2_img.shape[:2]
        rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        detected = results.pose_landmarks is not None
        confidence = self._calc_mean_confidence(results.pose_landmarks)

        return DetectionResult(
            landmarks=results.pose_landmarks,
            image_shape=(h, w),
            detected=detected,
            mean_confidence=confidence,
            cv2_image=cv2_img,
        )

    def _calc_mean_confidence(self, landmarks) -> float:
        """計算所有地標可見度平均值"""
        if landmarks is None:
            return 0.0
        visibilities = [lm.visibility for lm in landmarks.landmark]
        return float(np.mean(visibilities))

    def close(self):
        self._pose.close()


class DetectionResult:
    """姿勢偵測結果容器"""

    def __init__(self, landmarks, image_shape, detected, mean_confidence, cv2_image):
        self.landmarks      = landmarks       # mp pose_landmarks
        self.image_shape    = image_shape     # (h, w)
        self.detected       = detected        # bool
        self.mean_confidence = mean_confidence  # float 0–1
        self.cv2_image      = cv2_image       # BGR ndarray

    @property
    def h(self): return self.image_shape[0]

    @property
    def w(self): return self.image_shape[1]

    def get(self, idx: int) -> Optional[object]:
        """
        安全取得地標。
        若地標不存在或可見度過低，回傳 None。
        """
        if not self.detected or self.landmarks is None:
            return None
        if idx >= len(self.landmarks.landmark):
            return None
        lm = self.landmarks.landmark[idx]
        if lm.visibility < LANDMARK_VISIBILITY_THRESHOLD:
            return None
        return lm

    def px(self, idx: int) -> Optional[tuple]:
        """回傳地標像素座標 (x, y)，不存在時回傳 None"""
        lm = self.get(idx)
        if lm is None:
            return None
        return (int(lm.x * self.w), int(lm.y * self.h))

    def is_low_confidence(self) -> bool:
        return self.mean_confidence < 0.55

    def confidence_note(self) -> str:
        if not self.detected:
            return "未偵測到姿勢地標"
        if self.is_low_confidence():
            return f"地標信心偏低（{self.mean_confidence:.0%}），結果僅供參考"
        return f"地標信心良好（{self.mean_confidence:.0%}）"
