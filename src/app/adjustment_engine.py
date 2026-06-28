from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class AdjustmentSettings:
    temp: int = 0
    tint: int = 0
    exposure: float = 0.0
    contrast: int = 0
    highlights: int = 0
    shadows: int = 0
    whites: int = 0
    blacks: int = 0
    vibrance: int = 0
    saturation: int = 0
    clarity: int = 0
    dehaze: int = 0
    sharpen: int = 0


def calculate_resize(width: int, height: int, max_dimension: Optional[int]) -> tuple[int, int]:
    if not max_dimension or max_dimension <= 0:
        return width, height
    largest = max(width, height)
    if largest <= max_dimension:
        return width, height
    scale = max_dimension / largest
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _to_float(image_np: np.ndarray) -> np.ndarray:
    return np.clip(image_np.astype(np.float32) / 255.0, 0.0, 1.0)


def _to_uint8(image_float: np.ndarray) -> np.ndarray:
    return np.clip(image_float * 255.0, 0.0, 255.0).astype(np.uint8)


def _apply_temp_tint(img: np.ndarray, temp: int, tint: int) -> np.ndarray:
    if temp == 0 and tint == 0:
        return img
    result = img.copy()
    temp_amount = np.clip(temp / 100.0, -1.0, 1.0) * 0.08
    tint_amount = np.clip(tint / 100.0, -1.0, 1.0) * 0.06
    result[:, :, 0] *= 1.0 + temp_amount
    result[:, :, 2] *= 1.0 - temp_amount
    result[:, :, 1] *= 1.0 + tint_amount
    return np.clip(result, 0.0, 1.0)


def _apply_luminance_adjustments(img: np.ndarray, settings: AdjustmentSettings) -> np.ndarray:
    lab = cv2.cvtColor(_to_uint8(img), cv2.COLOR_RGB2LAB).astype(np.float32)
    l = lab[:, :, 0] / 255.0

    if settings.exposure:
        l = l * (2.0 ** settings.exposure)

    shadow_mask = np.square(np.clip(1.0 - l / 0.55, 0.0, 1.0))
    highlight_mask = np.square(np.clip((l - 0.45) / 0.55, 0.0, 1.0))
    white_mask = np.square(np.clip((l - 0.70) / 0.30, 0.0, 1.0))
    black_mask = np.square(np.clip(1.0 - l / 0.30, 0.0, 1.0))

    l += (settings.shadows / 100.0) * 0.28 * shadow_mask
    l += (settings.highlights / 100.0) * 0.28 * highlight_mask
    l += (settings.whites / 100.0) * 0.18 * white_mask
    l += (settings.blacks / 100.0) * 0.18 * black_mask

    if settings.contrast:
        amount = settings.contrast / 100.0 * 0.32
        l = (l - 0.5) * (1.0 + amount) + 0.5

    if settings.dehaze:
        amount = settings.dehaze / 100.0 * 0.12
        l = l + amount * (l - cv2.GaussianBlur(l, (0, 0), 16.0))

    lab[:, :, 0] = np.clip(l * 255.0, 0.0, 255.0)
    return _to_float(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB))


def _apply_vibrance_saturation(img: np.ndarray, vibrance: int, saturation: int) -> np.ndarray:
    hsv = cv2.cvtColor(_to_uint8(img), cv2.COLOR_RGB2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    if saturation:
        sat_factor = max(0.0, 1.0 + saturation / 100.0)
        s *= sat_factor
    if vibrance:
        vib = vibrance / 100.0
        s += (1.0 - s) * vib * 0.55 * (1.0 - s)
    hsv[:, :, 1] = np.clip(s * 255.0, 0.0, 255.0)
    return _to_float(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))


def _apply_clarity_sharpen(img: np.ndarray, clarity: int, sharpen: int) -> np.ndarray:
    result = img
    if clarity:
        amount = clarity / 100.0 * 0.22
        blur = cv2.GaussianBlur(result, (0, 0), 2.0)
        result = np.clip(result + amount * (result - blur), 0.0, 1.0)
    if sharpen:
        amount = sharpen / 100.0 * 0.35
        blur = cv2.GaussianBlur(result, (0, 0), 0.8)
        result = np.clip(result + amount * (result - blur), 0.0, 1.0)
    return result


def apply_adjustments(image_np: np.ndarray, settings: AdjustmentSettings) -> np.ndarray:
    img = _to_float(image_np)
    img = _apply_temp_tint(img, settings.temp, settings.tint)
    img = _apply_luminance_adjustments(img, settings)
    img = _apply_vibrance_saturation(img, settings.vibrance, settings.saturation)
    img = _apply_clarity_sharpen(img, settings.clarity, settings.sharpen)
    return _to_uint8(img)
