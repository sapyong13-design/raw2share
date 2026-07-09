import cv2
import numpy as np

from app.autocorrect import apply_recommended_adjustments


def test_outdoor_cloud_threshold_saturation_mask_does_not_speckle():
    img = np.zeros((160, 220, 3), dtype=np.uint8)
    img[:, :] = [104, 168, 215]
    hsv_cloud = np.zeros((75, 180, 3), dtype=np.uint8)
    hsv_cloud[:, :, 0] = 104
    for x in range(180):
        hsv_cloud[:, x, 1] = 28 + (x % 9)
    hsv_cloud[:, :, 2] = 202
    img[20:95, 20:200] = cv2.cvtColor(hsv_cloud, cv2.COLOR_HSV2RGB)
    img[105:, :] = [130, 95, 80]

    adjusted = apply_recommended_adjustments(
        img,
        is_raw=False,
        candidate_profile="balanced_pop",
        smart_strength="Event Balanced",
    )

    gray = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cloud = gray[25:90, 25:195]
    high_freq = cloud - cv2.GaussianBlur(cloud, (0, 0), 2.0)

    assert float(np.percentile(np.abs(high_freq), 95)) < 6.0
