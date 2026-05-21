#!/usr/bin/env python3
"""
camera_calibration.py
RGB 단안 카메라 + 평면(테이블) 가정 + 호모그래피 기반 좌표 변환.

- HOMOGRAPHY_PATH 환경변수로 호모그래피 .npy 파일 경로 지정
  (기본: ~/nero_calib.npy)
- 파일 없으면 더미 변환 + 경고 발생
- 캘리브레이션은 scripts/calibrate_homography.py 로 한 번 수행
"""

import os
import numpy as np


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
HOMOGRAPHY_PATH = os.environ.get(
    "HOMOGRAPHY_PATH",
    os.path.expanduser("~/nero_calib.npy"),
)

# 테이블 표면의 로봇 베이스 기준 z (미터)
# (호모그래피는 x, y 평면만 다루므로 z는 상수로 가정)
TABLE_Z = float(os.environ.get("TABLE_Z", "0.05"))


# ──────────────────────────────────────────────
# 호모그래피 로드 (모듈 import 시 1회)
# ──────────────────────────────────────────────
_H: "np.ndarray | None" = None


def _load_homography() -> bool:
    global _H
    if os.path.exists(HOMOGRAPHY_PATH):
        try:
            _H = np.load(HOMOGRAPHY_PATH)
            if _H.shape != (3, 3):
                print(f"[calib] ⚠️  잘못된 행렬 shape {_H.shape}, 무시")
                _H = None
                return False
            print(f"[calib] ✅ 호모그래피 로드: {HOMOGRAPHY_PATH}")
            return True
        except Exception as e:
            print(f"[calib] ⚠️  로드 실패: {e}")
            _H = None
            return False
    else:
        print(f"[calib] ⚠️  호모그래피 파일 없음: {HOMOGRAPHY_PATH}")
        print(f"[calib]    → scripts/calibrate_homography.py 먼저 실행하세요.")
        print(f"[calib]    → 지금은 더미 좌표 변환 사용 (실제 픽업 부정확)")
        return False


_loaded = _load_homography()


# ──────────────────────────────────────────────
# 메인 변환 함수
# ──────────────────────────────────────────────
def pixel_to_robot_xyz(
    px: float, py: float,
    img_w: int, img_h: int,
) -> dict:
    """
    픽셀 좌표 (px, py) → 로봇 베이스 좌표 (x, y, z) [미터].

    호모그래피가 로드돼 있으면 실제 변환, 없으면 더미.
    """
    if _H is not None:
        pt = np.array([px, py, 1.0], dtype=np.float64).reshape(3, 1)
        world = _H @ pt
        world /= world[2, 0]
        return {
            "x": round(float(world[0, 0]), 3),
            "y": round(float(world[1, 0]), 3),
            "z": TABLE_Z,
        }

    # 더미 변환 (캘리브레이션 전)
    nx = px / img_w
    ny = py / img_h
    return {
        "x": round(0.30 + (nx - 0.5) * 0.4, 3),
        "y": round((0.5 - ny) * 0.3, 3),
        "z": TABLE_Z,
        "_warning": "dummy_calibration",
    }


def reload_homography():
    """런타임에 호모그래피 재로드 (디버깅용)."""
    return _load_homography()


def is_calibrated() -> bool:
    return _H is not None
