import cv2
import numpy as np

from app.autocorrect import apply_recommended_adjustments


def test_outdoor_daylight_keeps_bright_sky_and_lifts_foreground_gently():
    img = np.zeros((180, 240, 3), dtype=np.uint8)
    img[:95, :] = [105, 165, 210]
    img[18:70, 25:115] = [238, 238, 232]
    img[34:92, 120:220] = [205, 210, 208]
    img[100:150, 25:215] = [142, 96, 82]
    img[150:, :] = [42, 46, 42]

    adjusted = apply_recommended_adjustments(
        img,
        is_raw=False,
        candidate_profile="balanced_pop",
        smart_strength="Event Balanced",
    )

    source_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    adjusted_gray = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)

    assert np.percentile(adjusted_gray, 95) >= np.percentile(source_gray, 95) - 18.0
    assert float(np.mean(adjusted_gray[100:150, 25:215])) >= float(np.mean(source_gray[100:150, 25:215])) - 2.0
