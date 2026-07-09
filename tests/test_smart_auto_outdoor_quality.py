import cv2
import numpy as np

from app.autocorrect import apply_recommended_adjustments


def test_outdoor_sky_adjustment_does_not_colorize_neutral_clouds():
    img = np.zeros((180, 240, 3), dtype=np.uint8)
    img[:, :] = [96, 158, 205]
    img[20:80, 20:110] = [238, 238, 232]
    img[35:95, 120:220] = [205, 210, 208]
    img[105:, :] = [130, 105, 84]

    adjusted = apply_recommended_adjustments(
        img,
        is_raw=False,
        candidate_profile="balanced_pop",
        smart_strength="Event Balanced",
    )

    source_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    adjusted_hsv = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV)
    cloud_mask = np.zeros(img.shape[:2], dtype=bool)
    cloud_mask[20:80, 20:110] = True
    cloud_mask[35:95, 120:220] = True

    source_sat = float(np.percentile(source_hsv[:, :, 1][cloud_mask], 95))
    adjusted_sat = float(np.percentile(adjusted_hsv[:, :, 1][cloud_mask], 95))

    assert adjusted_sat <= source_sat + 10.0
