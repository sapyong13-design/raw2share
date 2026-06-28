import cv2
import numpy as np
from app.image_analysis import analyze_image, recommend_adjustments

def auto_levels(image_np: np.ndarray, clip_hist_percent: float = 1.0) -> np.ndarray:
    """
    Applies auto-levels (histogram stretching) to normalize exposure.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_size = len(hist)
    
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))
        
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0
    
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
        
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
        
    if maximum_gray == minimum_gray:
        return image_np
        
    alpha = 255.0 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    result = cv2.convertScaleAbs(image_np, alpha=alpha, beta=beta)
    return result

def apply_clahe_color(image_np: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """
    Applies CLAHE on the LAB L-channel to enhance local contrast without affecting colors.
    """
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    result = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return result

def auto_exposure(image_np: np.ndarray, strength: float) -> np.ndarray:
    img_float = image_np.astype(np.float32)
    if strength > 0:
        factor = 1.0 + (strength * 0.35)
        img_float = img_float * factor
    elif strength < 0:
        factor = 1.0 + (strength * 0.25)
        img_float = img_float * factor
    return np.clip(img_float, 0, 255).astype(np.uint8)

def adjust_contrast(image_np: np.ndarray, strength: float) -> np.ndarray:
    factor = 1.0 + (strength * 0.35)
    img_float = image_np.astype(np.float32)
    img_float = (img_float - 127.5) * factor + 127.5
    return np.clip(img_float, 0, 255).astype(np.uint8)

def adjust_saturation(image_np: np.ndarray, strength: float) -> np.ndarray:
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    factor = 1.0 + (strength * 0.45)
    hsv[:, :, 1] = hsv[:, :, 1] * factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return result

def adjust_luminance_contrast(image_np: np.ndarray, amount: float) -> np.ndarray:
    """Apply gentle S-curve contrast only on luminance to keep colors stable."""
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32) / 255.0
    curved = l_float + amount * (l_float - 0.5) * (1.0 - np.abs(2.0 * l_float - 1.0))
    l_new = np.clip(curved * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l_new, a, b)), cv2.COLOR_LAB2RGB)

def raw_finishing_boost(image_np: np.ndarray) -> np.ndarray:
    """Visible but natural RAW finishing pass: shadows, contrast, vibrance, clarity."""
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32)
    p10, p50, p90 = np.percentile(l_float, [10, 50, 90])

    # Lift dark midtones/shadows without touching lamps and bright walls too much.
    shadow_mask = np.square(np.clip(1.0 - l_float / 180.0, 0.0, 1.0))
    mid_mask = np.clip(1.0 - np.abs(l_float - 105.0) / 115.0, 0.0, 1.0)
    shadow_lift = 14.0 if p10 < 55 else 8.0
    mid_lift = 8.0 if p50 < 120 else 3.0
    l_float = l_float + shadow_lift * shadow_mask + mid_lift * mid_mask

    # Preserve highlights; do not let ceiling lamps become colored blobs.
    highlight_mask = np.clip((l_float - 210.0) / 35.0, 0.0, 1.0)
    l_float = l_float - 5.0 * highlight_mask
    l_new = np.clip(l_float, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(cv2.merge((l_new, a, b)), cv2.COLOR_LAB2RGB)
    rgb = adjust_luminance_contrast(rgb, 0.12)

    # RAW often looks flat/desaturated after safe tone mapping; add restrained vibrance.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    s = hsv[:, :, 1]
    s = s + (255.0 - s) * 0.10 * (1.0 - s / 255.0)
    hsv[:, :, 1] = np.clip(s, 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Mild clarity: enough to look processed, not crunchy.
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.2)
    rgb = cv2.addWeighted(rgb, 1.10, blur, -0.10, 0)
    return np.clip(rgb, 0, 255).astype(np.uint8)




def _skin_mask_rgb(image_np: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    return cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32) / 255.0


def _safe_gray_world_white_balance(image_np: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return image_np
    rgb = image_np.astype(np.float32)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    neutral = (saturation < 72) & (value > 35) & (value < 235)
    if int(np.count_nonzero(neutral)) < image_np.shape[0] * image_np.shape[1] * 0.03:
        neutral = (value > 35) & (value < 230)
    if int(np.count_nonzero(neutral)) == 0:
        return image_np
    means = rgb[neutral].mean(axis=0)
    gray = float(means.mean())
    gains = gray / np.maximum(means, 1.0)
    gains = np.clip(gains, 0.92, 1.08)
    corrected = rgb * (1.0 + (gains - 1.0) * strength)
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _protect_lamp_highlights(image_np: np.ndarray) -> np.ndarray:
    rgb = image_np.astype(np.float32)
    y = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mask = np.clip((y - 214.0) / 26.0, 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)
    neutral = y[:, :, None]
    rgb = rgb * (1.0 - mask[:, :, None]) + neutral * mask[:, :, None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def apply_recommended_adjustments(
    image_np: np.ndarray,
    is_raw: bool = False,
    candidate_profile: str = "balanced_pop",
) -> np.ndarray:
    """Apply scene-aware Smart Auto adjustments from image analysis."""
    analysis = analyze_image(image_np, is_raw=is_raw)
    rec = recommend_adjustments(analysis)
    profile = {
        "natural_safe": {
            "exposure_bias": -0.01,
            "shadow_mult": 0.68,
            "highlight_mult": 1.10,
            "contrast_mult": 0.70,
            "vibrance_mult": 0.55,
            "clarity_mult": 0.55,
            "wb_mult": 0.70,
            "subject_boost": 0.00,
        },
        "people_bright": {
            "exposure_bias": 0.055,
            "shadow_mult": 0.95,
            "highlight_mult": 0.88,
            "contrast_mult": 0.62,
            "vibrance_mult": 0.62,
            "clarity_mult": 0.50,
            "wb_mult": 0.88,
            "subject_boost": 0.10,
        },
        "event_bright": {
            "exposure_bias": 0.075,
            "shadow_mult": 1.04,
            "highlight_mult": 0.95,
            "contrast_mult": 0.70,
            "vibrance_mult": 0.72,
            "clarity_mult": 0.55,
            "wb_mult": 0.90,
            "subject_boost": 0.13,
        },
        "event_pop": {
            "exposure_bias": 0.055,
            "shadow_mult": 0.92,
            "highlight_mult": 1.00,
            "contrast_mult": 0.95,
            "vibrance_mult": 1.05,
            "clarity_mult": 0.75,
            "wb_mult": 0.82,
            "subject_boost": 0.09,
        },
        "skin_safe": {
            "exposure_bias": 0.035,
            "shadow_mult": 0.82,
            "highlight_mult": 1.08,
            "contrast_mult": 0.52,
            "vibrance_mult": 0.40,
            "clarity_mult": 0.32,
            "wb_mult": 0.72,
            "subject_boost": 0.08,
        },
        "lamp_safe": {
            "exposure_bias": 0.015,
            "shadow_mult": 0.74,
            "highlight_mult": 1.55,
            "contrast_mult": 0.48,
            "vibrance_mult": 0.36,
            "clarity_mult": 0.30,
            "wb_mult": 0.55,
            "subject_boost": 0.04,
        },
        "low_light_people": {
            "exposure_bias": 0.105,
            "shadow_mult": 1.18,
            "highlight_mult": 1.08,
            "contrast_mult": 0.58,
            "vibrance_mult": 0.64,
            "clarity_mult": 0.42,
            "wb_mult": 0.86,
            "subject_boost": 0.16,
        },
        "highlight_safe": {
            "exposure_bias": -0.035,
            "shadow_mult": 0.62,
            "highlight_mult": 1.35,
            "contrast_mult": 0.58,
            "vibrance_mult": 0.42,
            "clarity_mult": 0.38,
            "wb_mult": 0.60,
            "subject_boost": 0.00,
        },
        "balanced_pop": {
            "exposure_bias": 0.025,
            "shadow_mult": 0.82,
            "highlight_mult": 1.00,
            "contrast_mult": 0.88,
            "vibrance_mult": 0.85,
            "clarity_mult": 0.80,
            "wb_mult": 0.82,
            "subject_boost": 0.04,
        },
    }.get(candidate_profile, {})

    # RAW needs lens/profile compensation. Camera/JPG images already include camera correction;
    # radial gain on JPG indoor shots creates colored lamp and wall artifacts.
    if is_raw and rec.correct_vignette:
        image_np = correct_raw_corner_shading(image_np, strength=0.82)

    skin_mask = _skin_mask_rgb(image_np)
    is_people_indoor = analysis.has_skin and analysis.has_subject and analysis.scene in {"indoor_balanced", "low_light", "backlit_high_contrast"}

    # Gentle gray-world WB from low-saturation neutrals, like a conservative Auto WB pass.
    wb_strength = (0.22 if is_people_indoor else 0.12) * profile.get("wb_mult", 1.0)
    image_np = _safe_gray_world_white_balance(image_np, wb_strength)

    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    lf = np.clip(l.astype(np.float32) / 255.0, 0.0, 1.0)

    # Subject-first exposure: center/skin receives more lift; highlights/lamp stay protected.
    h, w = lf.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    sigma_x, sigma_y = max(w * 0.34, 1.0), max(h * 0.34, 1.0)
    center_weight = np.exp(-(((xx - cx) ** 2) / (2 * sigma_x ** 2) + ((yy - cy) ** 2) / (2 * sigma_y ** 2))).astype(np.float32)
    subject_weight = np.clip(0.55 * center_weight + 0.75 * skin_mask, 0.0, 1.0)
    highlight_protect = 1.0 - np.clip((lf - 0.78) / 0.22, 0.0, 1.0) ** 2

    exposure = rec.exposure + profile.get("exposure_bias", 0.0)
    if is_people_indoor:
        exposure = max(exposure, 0.08 + profile.get("subject_boost", 0.0))
    if exposure:
        gain = 2.0 ** exposure
        lf = lf * (1.0 + (gain - 1.0) * highlight_protect * (0.55 + 0.45 * subject_weight))

    shadow_curve = lf * np.square(1.0 - lf)
    shadow_strength = rec.shadows * (0.55 if is_people_indoor else 0.75) * profile.get("shadow_mult", 1.0)
    lf = lf + shadow_strength * shadow_curve * (0.65 + 0.55 * subject_weight)

    # Highlight compression keeps pure white anchored, avoiding gray/color blobs on lamps.
    if rec.highlights < 0:
        shoulder = np.power(lf, 2.4) * (1.0 - lf)
        lf = lf + rec.highlights * profile.get("highlight_mult", 1.0) * 0.38 * shoulder

    lf = np.clip(lf, 0.0, 1.0)
    contrast = rec.contrast * (0.55 if is_people_indoor else 0.85) * profile.get("contrast_mult", 1.0)
    if contrast:
        lf = lf + contrast * 0.20 * (lf - 0.5) * (1.0 - np.abs(2.0 * lf - 1.0))

    l_new = np.clip(lf * 255.0, 0, 255).astype(np.uint8)
    # Keep chroma stable. Do not scale chroma aggressively on JPG event photos.
    ratio = np.clip((l_new.astype(np.float32) + 12.0) / (l.astype(np.float32) + 12.0), 0.82, 1.14)
    if is_people_indoor:
        ratio = 1.0 + (ratio - 1.0) * 0.35
    a_new = np.clip((a.astype(np.float32) - 128.0) * ratio + 128.0, 0, 255).astype(np.uint8)
    b_new = np.clip((b.astype(np.float32) - 128.0) * ratio + 128.0, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.merge((l_new, a_new, b_new)), cv2.COLOR_LAB2RGB)
    rgb = _protect_lamp_highlights(rgb)

    # Conservative vibrance. Skin pixels are mostly preserved.
    if rec.vibrance > 0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        s_ch = hsv[:, :, 1]
        vib = rec.vibrance * (0.55 if is_people_indoor else 0.85) * profile.get("vibrance_mult", 1.0)
        boosted = s_ch + (255.0 - s_ch) * vib * (1.0 - s_ch / 255.0)
        s_ch = boosted * (1.0 - 0.82 * skin_mask) + s_ch * (0.82 * skin_mask)
        hsv[:, :, 1] = np.clip(s_ch, 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if rec.denoise > 0:
        sigma = 8 + rec.denoise * 16
        rgb = cv2.bilateralFilter(rgb, 5, sigma, sigma)

    if rec.clarity > 0:
        blur = cv2.GaussianBlur(rgb, (0, 0), 1.0)
        amount = rec.clarity * (0.22 if is_people_indoor else 0.36) * profile.get("clarity_mult", 1.0)
        clarity_img = np.clip(rgb.astype(np.float32) + amount * (rgb.astype(np.float32) - blur.astype(np.float32)), 0, 255)
        rgb = (clarity_img * (1.0 - 0.75 * skin_mask[:, :, None]) + rgb.astype(np.float32) * (0.75 * skin_mask[:, :, None])).astype(np.uint8)

    if is_people_indoor:
        lab_skin = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_target = np.maximum(lab_skin[:, :, 0], l.astype(np.float32) + 5.0)
        skin_strength = np.clip(skin_mask * 0.55, 0.0, 0.55)
        lab_skin[:, :, 0] = lab_skin[:, :, 0] * (1.0 - skin_strength) + l_target * skin_strength
        rgb = cv2.cvtColor(np.clip(lab_skin, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    rgb = _protect_lamp_highlights(rgb)
    return np.clip(rgb, 0, 255).astype(np.uint8)
def _lab_luminance(image_np: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)


def _center_luminance_mean(image_np: np.ndarray) -> float:
    l_channel = _lab_luminance(image_np)
    h, w = l_channel.shape
    crop = l_channel[int(h * 0.22):int(h * 0.78), int(w * 0.18):int(w * 0.72)]
    return float(np.mean(crop)) if crop.size else float(np.mean(l_channel))


def _skin_luminance_mean(image_np: np.ndarray) -> float:
    skin = _skin_mask_rgb(image_np) > 0.35
    l_channel = _lab_luminance(image_np)
    if int(np.count_nonzero(skin)) < image_np.shape[0] * image_np.shape[1] * 0.01:
        return _center_luminance_mean(image_np)
    return float(np.mean(l_channel[skin]))


def _candidate_quality_score(original: np.ndarray, candidate: np.ndarray, is_raw: bool, profile: str) -> float:
    original_analysis = analyze_image(original, is_raw=is_raw)
    candidate_analysis = analyze_image(candidate, is_raw=is_raw)
    score = 100.0

    original_center = _center_luminance_mean(original)
    candidate_center = _center_luminance_mean(candidate)
    candidate_skin_l = _skin_luminance_mean(candidate)
    target_center = 143.0 if original_analysis.has_skin else 132.0
    target_skin = 146.0

    score -= abs(candidate_center - target_center) * 0.38
    if original_analysis.has_skin:
        score -= abs(candidate_skin_l - target_skin) * 0.32
        original_skin_l = _skin_luminance_mean(original)
        if candidate_skin_l < original_skin_l - 2.0:
            score -= (original_skin_l - candidate_skin_l) * 1.4

    center_lift = candidate_center - original_center
    if original_analysis.has_skin and center_lift < 6.0:
        score -= (6.0 - center_lift) * 2.4
    if center_lift > 24.0:
        score -= (center_lift - 24.0) * 1.7

    score -= max(0.0, candidate_analysis.highlight_clip_ratio - original_analysis.highlight_clip_ratio - 0.001) * 900.0
    score -= max(0.0, candidate_analysis.saturation_mean - 105.0) * 0.30
    score -= max(0.0, candidate_analysis.contrast - 72.0) * 0.20

    delta = float(np.mean(np.abs(candidate.astype(np.float32) - original.astype(np.float32))))
    max_delta = 24.0 if original_analysis.has_skin else 26.0
    if delta > max_delta:
        score -= (delta - max_delta) * 1.0

    if original_analysis.has_skin:
        skin = _skin_mask_rgb(candidate) > 0.35
        if int(np.count_nonzero(skin)):
            hsv = cv2.cvtColor(candidate, cv2.COLOR_RGB2HSV)
            skin_sat = float(np.mean(hsv[:, :, 1][skin]))
            if skin_sat > 98.0:
                score -= (skin_sat - 98.0) * 0.46

    # Small profile preference keeps results predictable when scores are close.
    score += {"event_bright": 3.0, "people_bright": 2.5, "event_pop": 1.8, "skin_safe": 1.4, "natural_safe": 1.0, "balanced_pop": 0.8, "lamp_safe": 0.7, "low_light_people": 0.6, "highlight_safe": 0.4}.get(profile, 0.0)
    return score


def _resize_for_scoring(image_np: np.ndarray, max_side: int = 900) -> np.ndarray:
    h, w = image_np.shape[:2]
    largest = max(h, w)
    if largest <= max_side:
        return image_np
    scale = max_side / float(largest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _profiles_for_analysis(analysis) -> list[str]:
    if analysis.scene == "low_light":
        return ["low_light_people", "people_bright", "skin_safe", "natural_safe", "lamp_safe"]
    if analysis.scene in {"indoor_balanced", "backlit_high_contrast"}:
        return ["event_bright", "people_bright", "event_pop", "skin_safe", "lamp_safe", "natural_safe", "balanced_pop"]
    if analysis.scene == "outdoor_highlight":
        return ["highlight_safe", "skin_safe", "natural_safe", "balanced_pop"]
    return ["balanced_pop", "event_pop", "skin_safe", "natural_safe", "people_bright"]


def select_best_smart_auto_candidate(image_np: np.ndarray, is_raw: bool = False) -> tuple[np.ndarray, str, float]:
    scoring_source = _resize_for_scoring(image_np)
    profiles = _profiles_for_analysis(analyze_image(scoring_source, is_raw=is_raw))
    if _center_luminance_mean(scoring_source) > 150.0:
        profiles = [profile for profile in profiles if profile in {"natural_safe", "skin_safe", "balanced_pop", "highlight_safe"}] or ["natural_safe"]
    best_profile = "natural_safe"
    best_score = -1e9
    for profile in profiles:
        preview_candidate = apply_recommended_adjustments(scoring_source, is_raw=is_raw, candidate_profile=profile)
        score = _candidate_quality_score(scoring_source, preview_candidate, is_raw, profile)
        if score > best_score:
            best_profile = profile
            best_score = score
    if best_score < 20.0 and _center_luminance_mean(scoring_source) > 145.0:
        return image_np.copy(), "original_safe", best_score
    best_image = apply_recommended_adjustments(image_np, is_raw=is_raw, candidate_profile=best_profile)
    return best_image, best_profile, best_score


def get_smart_auto_profile_name(image_np: np.ndarray, is_raw: bool = False) -> str:
    """Return the Smart Auto v4 profile selected for this image."""
    _, profile, _ = select_best_smart_auto_candidate(image_np, is_raw=is_raw)
    return profile


def correct_raw_corner_shading(image_np: np.ndarray, strength: float = 0.95) -> np.ndarray:
    """Correct RAW lens corner shading/vignetting with a soft radial gain."""
    height, width = image_np.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = radius / max(float(radius.max()), 1.0)

    # Smoothly starts near the mid-frame (0.35) and peaks at corners.
    mask = np.clip((radius - 0.35) / 0.65, 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)

    img = image_np.astype(np.float32)
    luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

    # Less aggressive highlight protection to ensure corner brightness matches sky
    highlight_protect = 1.0 - np.clip((luminance - 215.0) / 40.0, 0.0, 1.0)
    
    # Increase the gain multiplier to 0.48 to make correction much more effective
    gain = 1.0 + strength * 0.48 * mask * highlight_protect
    img *= gain[:, :, None]

    # Neutralize blue/purple color casts in corners more effectively
    corner = mask * highlight_protect
    img[:, :, 2] *= (1.0 - 0.045 * corner)  # Reduce blue channel cast
    img[:, :, 0] *= (1.0 + 0.02 * corner)   # Restore warm red channel channel

    return np.clip(img, 0, 255).astype(np.uint8)


def protect_clipped_highlights_only(image_np: np.ndarray) -> np.ndarray:
    """For already-bright outdoor RAW renders: avoid extra exposure/vibrance."""
    rgb = image_np.copy()
    r, g, b = cv2.split(rgb)
    y = 0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32)

    # Only near-white pixels are neutralized; normal colors remain untouched.
    desat = np.clip((y - 242.0) / 13.0, 0.0, 1.0)
    rgb = cv2.merge((
        ((1 - desat) * r.astype(np.float32) + desat * y).astype(np.uint8),
        ((1 - desat) * g.astype(np.float32) + desat * y).astype(np.uint8),
        ((1 - desat) * b.astype(np.float32) + desat * y).astype(np.uint8),
    ))

    # Tiny shadow lift only for very dark tree/person areas, not a global brightening.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, bb = cv2.split(lab)
    l_float = l.astype(np.float32)
    shadow = np.square(np.clip(1.0 - l_float / 115.0, 0.0, 1.0))
    l_float = l_float + 5.0 * shadow
    l_new = np.clip(l_float, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l_new, a, bb)), cv2.COLOR_LAB2RGB)

def raw_smart_lighting(image_np: np.ndarray) -> np.ndarray:
    """RAW-first Smart Auto: selective tone edits like Exposure/Shadows/Highlights."""
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32)
    mean_l = float(np.mean(l_float))
    p5, p25, p50, p75, p95 = np.percentile(l_float, [5, 25, 50, 75, 95])
    clipped_ratio = float(np.mean(l_float >= 250.0))

    # Outdoor/daylight scenes with clipped whites should not be brightened further.
    # This is the case for white shirts, concrete, clouds, or direct sun.
    if clipped_ratio > 0.035 or (p95 > 246.0 and mean_l > 112.0):
        return protect_clipped_highlights_only(image_np)

    highlight_mask = np.clip((l_float - 190.0) / 55.0, 0.0, 1.0)
    protected = 1.0 - np.square(highlight_mask)

    # Target brighter indoor RAW result. Protect lamps/bright walls while lifting people.
    target_mean = 145.0 if p95 < 220 else 138.0
    global_lift = np.clip(target_mean - mean_l, 8.0, 34.0)
    l_float = l_float + global_lift * protected

    # Lightroom/DxO-style selective tone: open shadows and midtones separately.
    shadow_mask = np.square(np.clip(1.0 - l_float / 145.0, 0.0, 1.0))
    mid_mask = np.clip(1.0 - np.abs(l_float - 115.0) / 95.0, 0.0, 1.0)
    shadow_lift = np.clip(34.0 - p25 * 0.22, 10.0, 28.0)
    mid_lift = np.clip(18.0 - p50 * 0.06, 7.0, 14.0)
    l_float = l_float + shadow_lift * shadow_mask * protected + mid_lift * mid_mask * protected

    # Pull strong highlights back a little instead of letting lamps bloom.
    highlight_pull = np.clip((p95 - 198.0) * 0.18, 0.0, 9.0)
    l_float = l_float - highlight_pull * highlight_mask
    l_float = np.clip(l_float, 0, 255)

    # Moderate S-curve on luminance: defined but not crunchy.
    lf = l_float / 255.0
    curve_amount = 0.10
    lf = lf + curve_amount * (lf - 0.5) * (1.0 - np.abs(2.0 * lf - 1.0))
    l_new = np.clip(lf * 255.0, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(cv2.merge((l_new, a, b)), cv2.COLOR_LAB2RGB)

    # Highlight rolloff only at near-white light sources.
    r, g, bb = cv2.split(rgb)
    y = 0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * bb.astype(np.float32)
    desat = np.clip((y - 235.0) / 20.0, 0.0, 1.0)
    rgb = cv2.merge((
        ((1 - desat) * r.astype(np.float32) + desat * y).astype(np.uint8),
        ((1 - desat) * g.astype(np.float32) + desat * y).astype(np.uint8),
        ((1 - desat) * bb.astype(np.float32) + desat * y).astype(np.uint8),
    ))

    # Visible but restrained vibrance; low-saturation pixels get more, saturated ones protected.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    s = hsv[:, :, 1]
    vibrance = 0.18
    s = s + (255.0 - s) * vibrance * (1.0 - s / 255.0)
    hsv[:, :, 1] = np.clip(s, 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Mild denoise for indoor high ISO and tiny clarity only.
    rgb = cv2.bilateralFilter(rgb, 5, 8, 8)
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    rgb = cv2.addWeighted(rgb, 1.06, blur, -0.06, 0)
    return np.clip(rgb, 0, 255).astype(np.uint8)

def sharpen(image_np: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return image_np
    kernel = np.array([
        [0, -strength * 0.25, 0],
        [-strength * 0.25, 1 + strength, -strength * 0.25],
        [0, -strength * 0.25, 0]
    ], dtype=np.float32)
    return cv2.filter2D(image_np, -1, kernel)

def denoise(image_np: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return image_np
    d = int(5 + strength * 2)
    sigma_color = 10 + strength * 15
    sigma_space = 10 + strength * 15
    return cv2.bilateralFilter(image_np, d, sigma_color, sigma_space)

def lift_shadows_and_protect_highlights(image_np: np.ndarray, shadow_lift: float, highlight_protect: float) -> np.ndarray:
    img_float = image_np.astype(np.float32)
    gray = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
    
    if shadow_lift > 0:
        shadow_mask = np.clip(1.0 - (gray / 255.0), 0.0, 1.0)
        lift_factor = 1.0 + (shadow_lift * 0.4)
        for c in range(3):
            boost = img_float[:, :, c] * lift_factor
            img_float[:, :, c] = img_float[:, :, c] * (1.0 - shadow_mask) + boost * shadow_mask
            
    if highlight_protect > 0:
        highlight_mask = np.clip(gray / 255.0, 0.0, 1.0)
        dim_factor = 1.0 - (highlight_protect * 0.2)
        for c in range(3):
            dimmed = img_float[:, :, c] * dim_factor
            img_float[:, :, c] = img_float[:, :, c] * (1.0 - highlight_mask) + dimmed * highlight_mask

    return np.clip(img_float, 0, 255).astype(np.uint8)

def smart_auto_enhance(image_np: np.ndarray, is_raw: bool = False) -> np.ndarray:
    """Choose the best Smart Auto candidate for this specific image."""
    best_image, _, _ = select_best_smart_auto_candidate(image_np, is_raw=is_raw)
    return best_image

def apply_autocorrect(image_np: np.ndarray, mode: str, is_raw: bool = False) -> np.ndarray:
    """
    Applies the chosen autocorrect mode to the image.
    """
    mode_clean = mode.strip().lower()
    
    # Default to smart auto if Smart Auto is selected
    if "smart" in mode_clean or "auto" in mode_clean:
        return smart_auto_enhance(image_np, is_raw=is_raw)
        
    if mode_clean == "off" or not mode_clean:
        return image_np

    img = image_np.copy()

    if mode_clean == "natural":
        img = auto_levels(img, clip_hist_percent=0.8)
        img = apply_clahe_color(img, clip_limit=1.5, grid_size=8)
        img = adjust_saturation(img, 0.15)
        img = sharpen(img, 0.2)

    elif mode_clean == "bright":
        img = auto_levels(img, clip_hist_percent=1.5)
        img = auto_exposure(img, 0.3)
        img = lift_shadows_and_protect_highlights(img, shadow_lift=0.0, highlight_protect=0.6)

    elif mode_clean == "vivid":
        img = auto_levels(img, clip_hist_percent=1.0)
        img = apply_clahe_color(img, clip_limit=2.5, grid_size=8)
        img = adjust_contrast(img, 0.2)
        img = adjust_saturation(img, 0.45)

    elif mode_clean == "low light":
        img = denoise(img, 0.4)
        img = lift_shadows_and_protect_highlights(img, shadow_lift=0.6, highlight_protect=0.0)
        img = auto_levels(img, clip_hist_percent=2.0)
        img = sharpen(img, 0.35)

    return img
