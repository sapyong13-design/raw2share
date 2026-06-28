import cv2
import numpy as np

from app.autocorrect import get_smart_auto_decision, get_smart_auto_profile_name, save_smart_auto_contact_sheet, smart_auto_enhance, select_best_smart_auto_candidate


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


def test_smart_auto_decision_and_contact_sheet(tmp_path):
    image = np.full((80, 120, 3), [170, 155, 130], dtype=np.uint8)
    image[25:65, 35:70] = [185, 128, 92]
    decision = get_smart_auto_decision(image, is_raw=False)
    assert decision["profile"]
    assert isinstance(decision["score"], float)
    assert decision["candidates"]

    sheet_path = tmp_path / "candidates.jpg"
    result = save_smart_auto_contact_sheet(image, str(sheet_path), is_raw=False)
    assert result == str(sheet_path)
    assert sheet_path.exists()
    assert sheet_path.stat().st_size > 0


def test_smart_auto_strength_and_batch_context_are_accepted():
    image = np.full((80, 120, 3), [160, 145, 120], dtype=np.uint8)
    image[25:65, 35:70] = [180, 125, 90]
    context = {"dominant_scene": "indoor_balanced", "target_center": 138.0, "target_skin": 144.0, "sample_count": 4}
    result, profile, score = select_best_smart_auto_candidate(image, is_raw=False, batch_context=context, smart_strength="Event Bright")
    assert result.shape == image.shape
    assert profile
    assert isinstance(score, float)
    decision = get_smart_auto_decision(image, is_raw=False, batch_context=context, smart_strength="Event Bright")
    assert decision["profile"] == profile
