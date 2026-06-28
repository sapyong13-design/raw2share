import cv2
import numpy as np

from app.autocorrect import get_smart_auto_profile_name, smart_auto_enhance, select_best_smart_auto_candidate


def _center_luminance(image):
    l_channel = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    h, w = l_channel.shape
    crop = l_channel[int(h * 0.25):int(h * 0.75), int(w * 0.2):int(w * 0.7)]
    return float(np.mean(crop))


def test_smart_auto_candidate_lifts_people_region_without_lamp_color_blob():
    image = np.full((180, 240, 3), [176, 164, 135], dtype=np.uint8)
    image[55:150, 65:115] = [105, 78, 62]
    image[70:145, 115:150] = [184, 130, 96]
    image[25:55, 185:215] = [255, 255, 255]

    result, profile, score = select_best_smart_auto_candidate(image, is_raw=False)

    assert profile in {"natural_safe", "people_bright", "event_bright", "event_pop", "skin_safe", "lamp_safe", "low_light_people", "highlight_safe", "balanced_pop", "original_safe"}
    assert score > 0
    assert get_smart_auto_profile_name(image, is_raw=False) == profile
    skin_l = cv2.cvtColor(result[70:145, 115:150], cv2.COLOR_RGB2LAB)[:, :, 0]
    assert 118.0 <= float(np.mean(skin_l)) <= 170.0

    lamp = result[25:55, 185:215].astype(np.int16)
    assert float(np.mean(np.max(lamp, axis=2) - np.min(lamp, axis=2))) < 10.0


def test_smart_auto_enhance_returns_uint8_same_shape():
    image = np.full((64, 96, 3), 120, dtype=np.uint8)
    result = smart_auto_enhance(image, is_raw=False)
    assert result.shape == image.shape
    assert result.dtype == np.uint8
