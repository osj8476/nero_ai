#!/usr/bin/env python3
"""
calibrate_homography.py
RGB 단안 카메라 + 테이블 평면 호모그래피 캘리브레이션.

사용법:
    1) 테이블 위에 4점을 표시 (마스킹테이프 십자 등). 최소 4점, 6~8점 권장.
    2) 로봇을 각 점 위로 이동시키며 (x, y) 미터 좌표를 기록.
       → 아래 ROBOT_POINTS 리스트에 입력.
    3) python3 calibrate_homography.py 실행.
    4) 카메라 화면이 뜨면 ROBOT_POINTS 순서대로 각 점을 마우스 클릭.
    5) 'c' 키로 확인 → ~/nero_calib.npy 저장.
       'r' 키로 리셋, 'q' 키로 종료.

⚠️ 클릭 순서가 ROBOT_POINTS 순서와 정확히 일치해야 함!
"""

import os
import sys
import cv2
import numpy as np


# ──────────────────────────────────────────────
# ★ 사용자 입력 영역 ★
# ──────────────────────────────────────────────
# 로봇 베이스 기준 (x, y) 좌표 [미터].
# 테이블 위의 4점 이상을 실제로 측정해서 입력.
# 예시 (실제 측정값으로 교체!):
#   - 로봇 정면 30cm, 좌 20cm  → (0.30,  0.20)
#   - 로봇 정면 30cm, 우 20cm  → (0.30, -0.20)
#   - 로봇 정면 50cm, 우 20cm  → (0.50, -0.20)
#   - 로봇 정면 50cm, 좌 20cm  → (0.50,  0.20)
ROBOT_POINTS = [
    (0.30,  0.20),   # 클릭 1번째
    (0.30, -0.20),   # 클릭 2번째
    (0.50, -0.20),   # 클릭 3번째
    (0.50,  0.20),   # 클릭 4번째
]

CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "0")
OUTPUT_PATH = os.environ.get(
    "HOMOGRAPHY_PATH",
    os.path.expanduser("~/nero_calib.npy"),
)


# ──────────────────────────────────────────────
# 마우스 클릭 수집
# ──────────────────────────────────────────────
clicked_points = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < len(ROBOT_POINTS):
            clicked_points.append((x, y))
            idx = len(clicked_points) - 1
            print(f"[click {idx+1}] 픽셀 ({x}, {y}) "
                  f"→ 로봇 {ROBOT_POINTS[idx]}")


def draw_overlay(frame):
    """클릭된 점과 다음 안내 표시."""
    for i, (x, y) in enumerate(clicked_points):
        cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
        cv2.circle(frame, (x, y), 10, (0, 0, 0), 2)
        cv2.putText(frame, f"{i+1}: {ROBOT_POINTS[i]}", (x+12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    n = len(clicked_points)
    if n < len(ROBOT_POINTS):
        msg = (f"클릭 {n+1}/{len(ROBOT_POINTS)}: "
               f"로봇좌표 {ROBOT_POINTS[n]} 위치를 화면에서 클릭")
        color = (0, 200, 255)
    else:
        msg = "완료! 'c'=저장  'r'=리셋  'q'=종료"
        color = (0, 255, 0)
    cv2.putText(frame, msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def compute_and_save():
    if len(clicked_points) < 4:
        print("⚠️  최소 4점 필요")
        return False

    src = np.array(clicked_points, dtype=np.float64)   # 픽셀
    dst = np.array(ROBOT_POINTS, dtype=np.float64)     # 로봇 미터

    # 4점이면 직접 계산, 그 이상이면 RANSAC
    if len(src) == 4:
        H = cv2.getPerspectiveTransform(
            src.astype(np.float32),
            dst.astype(np.float32),
        ).astype(np.float64)
    else:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            print("⚠️  호모그래피 계산 실패")
            return False

    # 검증
    print("\n=== 캘리브레이션 검증 ===")
    errors = []
    for (px, py), (rx, ry) in zip(clicked_points, ROBOT_POINTS):
        pt = np.array([px, py, 1.0]).reshape(3, 1)
        proj = H @ pt
        proj /= proj[2, 0]
        err = np.hypot(proj[0, 0] - rx, proj[1, 0] - ry)
        errors.append(err)
        print(f"  픽셀({px:4d},{py:4d}) → 예측({proj[0,0]:+.3f},{proj[1,0]:+.3f}) "
              f"실제({rx:+.3f},{ry:+.3f}) 오차={err*1000:.1f}mm")
    mean_err = float(np.mean(errors))
    print(f"평균 오차: {mean_err*1000:.1f}mm")
    if mean_err > 0.02:
        print("⚠️  오차가 큽니다 (>20mm). 측정/클릭 정확도 확인 필요.")

    np.save(OUTPUT_PATH, H)
    print(f"\n✅ 저장 완료: {OUTPUT_PATH}")
    print(f"행렬:\n{H}")
    return True


def main():
    if len(ROBOT_POINTS) < 4:
        print("❌ ROBOT_POINTS에 최소 4점 입력 필요")
        sys.exit(1)

    source = int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ 카메라 열기 실패: {CAMERA_SOURCE}")
        sys.exit(1)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_mouse)

    print(f"카메라 소스: {CAMERA_SOURCE}")
    print(f"수집할 점 개수: {len(ROBOT_POINTS)}")
    print(f"순서대로 클릭하세요:")
    for i, p in enumerate(ROBOT_POINTS, 1):
        print(f"  {i}. 로봇좌표 ({p[0]:+.3f}, {p[1]:+.3f}) m")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        draw_overlay(display)
        cv2.imshow("Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            clicked_points.clear()
            print("리셋")
        elif key == ord('c'):
            if compute_and_save():
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
