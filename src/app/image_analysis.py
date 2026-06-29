from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class ImageAnalysis:
    mean_luminance: float
    p5_luminance: float
    p50_luminance: float
    p95_luminance: float
    highlight_clip_ratio: float
    shadow_ratio: float
    saturation_mean: float
    contrast: float
    noise_level: float
    vignette_strength: float
    exposure: str
    scene: str
    is_raw: bool
    skin_ratio: float
    has_skin: bool
    subject_ratio: float
    has_subject: bool
    face_count: int
    face_ratio: float
    has_face: bool
    reasoning: str


@dataclass(frozen=True)
class AdjustmentRecommendation:
    exposure: float
    highlights: float
    shadows: float
    contrast: float
    vibrance: float
    clarity: float
    denoise: float
    protect_highlights: bool
    correct_vignette: bool
    protect_skin: bool
    reasoning: str


def classify_scene(
    *,
    mean_luminance: float,
    highlight_clip_ratio: float,
    shadow_ratio: float,
    saturation_mean: float,
    is_raw: bool,
) -> str:
    if highlight_clip_ratio > 0.035 or mean_luminance > 210 or (mean_luminance > 165 and highlight_clip_ratio > 0.015):
        return "outdoor_highlight"
    if mean_luminance < 70 or shadow_ratio > 0.55:
        return "low_light"
    if highlight_clip_ratio > 0.01 and shadow_ratio > 0.25:
        return "backlit_high_contrast"
    if mean_luminance > 145 and saturation_mean > 45:
        return "outdoor_balanced"
    return "indoor_balanced"


def _estimate_noise(gray: np.ndarray) -> float:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.std(laplacian))


def _estimate_vignette(luminance: np.ndarray) -> float:
    height, width = luminance.shape
    center_crop = luminance[height // 4: height * 3 // 4, width // 4: width * 3 // 4]
    corner_size_h = max(1, height // 8)
    corner_size_w = max(1, width // 8)
    corners = np.concatenate([
        luminance[:corner_size_h, :corner_size_w].ravel(),
        luminance[:corner_size_h, -corner_size_w:].ravel(),
        luminance[-corner_size_h:, :corner_size_w].ravel(),
        luminance[-corner_size_h:, -corner_size_w:].ravel(),
    ])
    center_mean = float(np.mean(center_crop))
    corner_mean = float(np.mean(corners))
    if center_mean <= 1:
        return 0.0
    return max(0.0, min(1.0, (center_mean - corner_mean) / center_mean))


def _detect_faces(gray: np.ndarray) -> tuple[int, float]:
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return 0, 0.0
        scale = 1.0
        small = gray
        largest = max(gray.shape)
        if largest > 900:
            scale = 900.0 / largest
            small = cv2.resize(gray, (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        faces = cascade.detectMultiScale(small, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        if len(faces) == 0:
            return 0, 0.0
        area = 0.0
        for _, _, width, height in faces:
            area += float(width * height) / max(scale * scale, 1e-6)
        return int(len(faces)), float(min(1.0, area / float(gray.shape[0] * gray.shape[1])))
    except Exception:
        return 0, 0.0

_analysis_cache = {}

def analyze_image(image_np: np.ndarray, is_raw: bool = False) -> ImageAnalysis:
    key = (id(image_np), image_np.shape, is_raw, int(image_np.flat[0]), int(image_np.flat[-1]), float(image_np.flat[::2000].sum()) if image_np.size > 2000 else float(image_np.sum()))
    if key in _analysis_cache:
        return _analysis_cache[key]
        
    rgb = image_np[:, :, :3]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luminance = lab[:, :, 0].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)

    mean_luminance = float(np.mean(luminance))
    p5, p50, p95 = np.percentile(luminance, [5, 50, 95])
    highlight_clip_ratio = float(np.mean(luminance >= 250.0))
    shadow_ratio = float(np.mean(luminance <= 45.0))
    saturation_mean = float(np.mean(saturation))
    contrast = float(np.std(luminance))
    noise_level = _estimate_noise(gray)
    vignette_strength = _estimate_vignette(luminance)

    if highlight_clip_ratio > 0.035 or mean_luminance > 175:
        exposure = "overexposed"
    elif mean_luminance < 75 or shadow_ratio > 0.55:
        exposure = "underexposed"
    else:
        exposure = "balanced"

    scene = classify_scene(
        mean_luminance=mean_luminance,
        highlight_clip_ratio=highlight_clip_ratio,
        shadow_ratio=shadow_ratio,
        saturation_mean=saturation_mean,
        is_raw=is_raw,
    )

    # Skin detection (YCrCb)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    skin_ratio = float(np.mean(skin_mask > 0))
    has_skin = bool(skin_ratio > 0.01)

    # Subject detection (Center region)
    h, w = gray.shape
    cy_min, cy_max = int(h * 0.2), int(h * 0.8)
    cx_min, cx_max = int(w * 0.2), int(w * 0.8)
    center_region = np.zeros_like(gray, dtype=bool)
    center_region[cy_min:cy_max, cx_min:cx_max] = True

    overall_mean = np.mean(gray)
    center_deviation = np.abs(gray.astype(np.float32) - overall_mean) > 25
    subject_mask = (center_deviation | (skin_mask > 0)) & center_region
    subject_ratio = float(np.mean(subject_mask))
    face_count, face_ratio = _detect_faces(gray)
    has_face = bool(face_count > 0)
    if has_face:
        subject_ratio = max(subject_ratio, min(0.35, face_ratio * 4.0))
    has_subject = bool(subject_ratio > 0.02 or has_face)

    reasons = []
    reasons.append(f"Scene: {scene} ({exposure}).")
    if highlight_clip_ratio > 0.03:
        reasons.append("Highlight clipping / bright spots detected.")
    if skin_ratio > 0.05:
        reasons.append(f"Skin tones present ({skin_ratio:.1%}).")
    if has_face:
        reasons.append(f"Face detected ({face_count}).")
    if has_subject:
        reasons.append(f"Subject in center ({subject_ratio:.1%}).")
    if vignette_strength > 0.10:
        reasons.append(f"Vignette detected ({vignette_strength:.2f}).")
    reasoning = " ".join(reasons)

    res = ImageAnalysis(
        mean_luminance=mean_luminance,
        p5_luminance=float(p5),
        p50_luminance=float(p50),
        p95_luminance=float(p95),
        highlight_clip_ratio=highlight_clip_ratio,
        shadow_ratio=shadow_ratio,
        saturation_mean=saturation_mean,
        contrast=contrast,
        noise_level=noise_level,
        vignette_strength=vignette_strength,
        exposure=exposure,
        scene=scene,
        is_raw=is_raw,
        skin_ratio=skin_ratio,
        has_skin=has_skin,
        subject_ratio=subject_ratio,
        has_subject=has_subject,
        face_count=face_count,
        face_ratio=face_ratio,
        has_face=has_face,
        reasoning=reasoning,
    )
    if len(_analysis_cache) > 50:
        _analysis_cache.clear()
    _analysis_cache[key] = res
    return res


def recommend_adjustments(analysis: ImageAnalysis) -> AdjustmentRecommendation:
    reasons = []
    protect_skin = analysis.has_skin

    # Adaptive base exposure boost based on image mean luminance (target L is 115)
    target_lum = 115.0
    lum_diff = target_lum - analysis.mean_luminance
    max_exp_boost = 0.55 if analysis.is_raw else 0.32
    adaptive_exposure = float(np.clip(lum_diff / 100.0, -0.25, max_exp_boost))

    if analysis.scene == "outdoor_highlight":
        # Pull down exposure more aggressively if highlights are heavily clipped
        if analysis.highlight_clip_ratio > 0.05:
            exposure_adj = -0.22
        elif analysis.highlight_clip_ratio > 0.01:
            exposure_adj = -0.10
        else:
            exposure_adj = 0.0
        
        highlights_adj = -0.70  # Strong highlight pull-down
        shadows_adj = 0.25
        contrast_adj = 0.15
        vibrance_adj = 0.05
        clarity_adj = 0.08
        denoise_adj = 0.0
        reasons.append("Outdoor highlight scene: highlight recovery applied, exposure kept down.")
    elif analysis.scene == "low_light":
        exposure_adj = adaptive_exposure + 0.15
        highlights_adj = -0.25
        shadows_adj = 0.55 if analysis.is_raw else 0.35
        contrast_adj = 0.16
        vibrance_adj = 0.12
        clarity_adj = 0.10
        denoise_adj = 0.40 if analysis.noise_level > 8 else 0.20
        reasons.append("Low light scene: shadow lift and light denoise applied.")
    elif analysis.scene == "backlit_high_contrast":
        exposure_adj = adaptive_exposure + 0.05
        highlights_adj = -0.55
        shadows_adj = 0.45
        contrast_adj = 0.08
        vibrance_adj = 0.12
        clarity_adj = 0.08
        denoise_adj = 0.10
        reasons.append("Backlit scene: balanced highlight compression and shadow fill.")
    elif analysis.scene == "outdoor_balanced":
        exposure_adj = adaptive_exposure
        highlights_adj = -0.25
        shadows_adj = 0.20
        contrast_adj = 0.15
        vibrance_adj = 0.14
        clarity_adj = 0.08
        denoise_adj = 0.0
        reasons.append("Outdoor balanced: slight contrast and color boost.")
    else:  # indoor_balanced
        exposure_adj = max(0.0, adaptive_exposure) + (0.06 if analysis.has_subject else 0.03)
        highlights_adj = -0.18
        shadows_adj = 0.28
        contrast_adj = 0.08
        vibrance_adj = 0.06
        clarity_adj = 0.035
        denoise_adj = 0.15 if analysis.is_raw and analysis.noise_level > 12 else 0.0
        reasons.append("Indoor balanced scene: standard baseline enhancement.")

    if protect_skin:
        reasons.append("Skin tones detected; protecting skin colors.")

    if analysis.has_subject:
        reasons.append("Subject detected; adjusting local contrast/exposure.")
        if analysis.mean_luminance < 140:
            shadows_adj = min(0.65, shadows_adj + 0.12)

    correct_vignette = analysis.vignette_strength > 0.05
    if correct_vignette:
        reasons.append("Vignette correction recommended.")

    reasoning = " ".join(reasons)

    return AdjustmentRecommendation(
        exposure=exposure_adj,
        highlights=highlights_adj,
        shadows=shadows_adj,
        contrast=contrast_adj,
        vibrance=vibrance_adj,
        clarity=clarity_adj,
        denoise=denoise_adj,
        protect_highlights=True,
        correct_vignette=correct_vignette,
        protect_skin=protect_skin,
        reasoning=reasoning,
    )
