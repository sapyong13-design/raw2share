import numpy as np

from app.adjustment_engine import AdjustmentSettings, apply_adjustments, calculate_resize


def gradient_image():
    x = np.linspace(0, 255, 64, dtype=np.uint8)
    img = np.tile(x, (64, 1))
    return np.stack([img, img, img], axis=2)


def test_exposure_brightens_image():
    image = gradient_image()
    adjusted = apply_adjustments(image, AdjustmentSettings(exposure=1.0))
    assert adjusted.mean() > image.mean()


def test_saturation_minus_100_grayscale():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 0] = 200
    image[:, :, 1] = 50
    image[:, :, 2] = 20
    adjusted = apply_adjustments(image, AdjustmentSettings(saturation=-100))
    assert np.allclose(adjusted[:, :, 0], adjusted[:, :, 1], atol=2)
    assert np.allclose(adjusted[:, :, 1], adjusted[:, :, 2], atol=2)


def test_highlights_negative_darkens_bright_pixels_more_than_shadows():
    image = gradient_image()
    adjusted = apply_adjustments(image, AdjustmentSettings(highlights=-80))
    bright_delta = int(image[0, -1, 0]) - int(adjusted[0, -1, 0])
    dark_delta = int(image[0, 5, 0]) - int(adjusted[0, 5, 0])
    assert bright_delta > dark_delta


def test_calculate_resize_preserves_aspect_and_no_upscale():
    assert calculate_resize(6000, 4000, 3000) == (3000, 2000)
    assert calculate_resize(1200, 800, 3000) == (1200, 800)
    assert calculate_resize(4000, 6000, 3000) == (2000, 3000)
