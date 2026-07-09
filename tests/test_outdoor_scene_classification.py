import cv2
import numpy as np

from app.image_analysis import analyze_image


def test_outdoor_building_with_false_skin_tones_classifies_as_outdoor():
    img = np.zeros((220, 320, 3), dtype=np.uint8)
    img[:100, :] = [98, 160, 210]
    cv2.circle(img, (90, 45), 42, (232, 232, 226), -1)
    cv2.circle(img, (210, 55), 55, (195, 202, 204), -1)
    img[100:165, 35:285] = [184, 86, 72]
    img[118:180, 70:250] = [205, 185, 148]
    img[180:, :] = [48, 52, 48]

    analysis = analyze_image(img, is_raw=False)

    assert analysis.scene == "outdoor_balanced"
