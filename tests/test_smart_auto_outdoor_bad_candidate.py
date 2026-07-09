import cv2
import numpy as np

from app.autocorrect import _candidate_quality_score


def _outdoor_building_fixture():
    img = np.zeros((220, 320, 3), dtype=np.uint8)
    img[:100, :] = [98, 160, 210]
    cv2.circle(img, (90, 45), 42, (232, 232, 226), -1)
    cv2.circle(img, (210, 55), 55, (195, 202, 204), -1)
    img[100:165, 35:285] = [184, 86, 72]
    img[118:180, 70:250] = [205, 185, 148]
    img[180:, :] = [48, 52, 48]
    return img


def test_outdoor_scoring_rejects_oversaturated_clipped_candidate():
    original = _outdoor_building_fixture()
    hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 2.1 + 45.0, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.25 + 28.0, 0, 255)
    bad = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    score = _candidate_quality_score(original, bad, is_raw=False, profile="event_pop")

    assert score < 90.0
