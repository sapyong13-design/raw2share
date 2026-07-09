import cv2
import numpy as np

from app.autocorrect import apply_recommended_adjustments


def test_outdoor_clouds_do_not_get_halo_ringing_or_dark_fill():
    img = np.zeros((180, 240, 3), dtype=np.uint8)
    img[:, :] = [95, 155, 205]
    cv2.circle(img, (70, 55), 35, (238, 238, 232), -1)
    cv2.circle(img, (150, 55), 42, (210, 213, 208), -1)
    img[105:, :] = [120, 90, 75]

    adjusted = apply_recommended_adjustments(
        img,
        is_raw=False,
        candidate_profile="balanced_pop",
        smart_strength="Event Balanced",
    )

    source_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    adjusted_gray = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY)
    cloud_mask = source_gray > 200
    ring = cv2.dilate(cloud_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(bool) & ~cloud_mask

    assert np.percentile(adjusted_gray[cloud_mask], 5) >= np.percentile(source_gray[cloud_mask], 5) - 24.0
    assert np.percentile(adjusted_gray[ring], 95) <= np.percentile(source_gray[cloud_mask], 5) + 8.0
