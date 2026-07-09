import cv2
import numpy as np

from app.autocorrect import get_smart_auto_decision


def test_outdoor_building_false_skin_scene_scores_safe_candidate_high():
    img = np.zeros((220, 320, 3), dtype=np.uint8)
    img[:100, :] = [98, 160, 210]
    cv2.circle(img, (90, 45), 42, (232, 232, 226), -1)
    cv2.circle(img, (210, 55), 55, (195, 202, 204), -1)
    img[100:165, 35:285] = [184, 86, 72]
    img[118:180, 70:250] = [205, 185, 148]
    img[180:, :] = [48, 52, 48]

    decision = get_smart_auto_decision(img, is_raw=False, smart_strength="Event Balanced")

    assert decision["profile"] in {"balanced_pop", "event_pop", "natural_safe", "skin_safe"}
    assert decision["score"] >= 96.0
