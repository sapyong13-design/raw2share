import cv2
import numpy as np

from app.autocorrect import apply_recommended_adjustments


def test_outdoor_smart_auto_does_not_inject_texture_into_smooth_clouds():
    img = np.zeros((180, 240, 3), dtype=np.uint8)
    img[:, :] = [110, 170, 215]
    img[15:95, 25:220] = [210, 214, 212]
    rng = np.random.default_rng(7)
    cloud_noise = rng.integers(-3, 4, size=(80, 195, 1), dtype=np.int16)
    noisy_cloud = np.clip(img[15:95, 25:220].astype(np.int16) + cloud_noise, 0, 255).astype(np.uint8)
    img[15:95, 25:220] = noisy_cloud
    img[35:75, 120:210] = [178, 186, 188]
    img[105:, :] = [125, 95, 80]

    adjusted = apply_recommended_adjustments(
        img,
        is_raw=False,
        candidate_profile="balanced_pop",
        smart_strength="Event Balanced",
    )

    gray_source = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_adjusted = cv2.cvtColor(adjusted, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth_cloud = np.zeros(img.shape[:2], dtype=bool)
    smooth_cloud[18:92, 30:215] = True

    source_texture = cv2.Laplacian(gray_source, cv2.CV_32F)[smooth_cloud]
    adjusted_texture = cv2.Laplacian(gray_adjusted, cv2.CV_32F)[smooth_cloud]

    assert float(np.percentile(np.abs(adjusted_texture), 95)) <= float(np.percentile(np.abs(source_texture), 95)) * 1.25 + 2.0
