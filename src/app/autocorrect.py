import cv2
import numpy as np
import os
from app.image_analysis import analyze_image, recommend_adjustments

def extract_metadata_stats(image_path: str) -> dict:
    """
    Extract focal length, aperture, ISO, and lens info from RAW or JPG file.
    """
    stats = {
        'focal_length': 35.0,  # default mid-range
        'aperture': 5.6,       # default
        'iso': 100.0,          # default
        'lens_model': 'Unknown',
        'is_raw': False
    }
    if not image_path or not os.path.exists(image_path):
        return stats
        
    _, ext = os.path.splitext(image_path.lower())
    if ext in ('.jpg', '.jpeg'):
        try:
            from PIL import Image
            img = Image.open(image_path)
            exif = img._getexif()
            if exif:
                if 37386 in exif:
                    val = exif[37386]
                    stats['focal_length'] = float(val[0]/val[1]) if isinstance(val, tuple) else float(val)
                if 33437 in exif:
                    val = exif[33437]
                    stats['aperture'] = float(val[0]/val[1]) if isinstance(val, tuple) else float(val)
                if 34855 in exif:
                    stats['iso'] = float(exif[34855])
                if 42036 in exif:
                    stats['lens_model'] = str(exif[42036])
        except Exception:
            pass
    else:
        # RAW file via rawpy
        try:
            import rawpy
            with rawpy.imread(image_path) as raw:
                stats['is_raw'] = True
                if raw.lens and raw.lens.model:
                    stats['lens_model'] = raw.lens.model
                if raw.other:
                    if hasattr(raw.other, 'focal_length') and raw.other.focal_length:
                        stats['focal_length'] = float(raw.other.focal_length)
                    if hasattr(raw.other, 'aperture') and raw.other.aperture:
                        stats['aperture'] = float(raw.other.aperture)
                    if hasattr(raw.other, 'iso_speed') and raw.other.iso_speed:
                        stats['iso'] = float(raw.other.iso_speed)
        except Exception:
            pass
    return stats

def correct_lens_artifacts(image_np: np.ndarray, metadata: dict) -> np.ndarray:
    """
    EXIF-Guided Lens Profile Database:
    Corrects vignetting and chromatic aberration based on focal length and aperture.
    """
    focal = metadata.get('focal_length', 35.0)
    aperture = metadata.get('aperture', 5.6)
    is_raw = metadata.get('is_raw', False)
    
    height, width = image_np.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = radius / max(float(radius.max()), 1.0)
    
    vignette_mult = 1.0
    if focal < 24.0:
        vignette_mult *= 1.45
    elif focal > 70.0:
        vignette_mult *= 0.45
        
    if aperture < 4.0:
        vignette_mult *= 1.0 + max(0.0, (4.0 - aperture) * 0.3)
        
    mask = np.clip((radius - 0.35) / 0.65, 0.0, 1.0)
    mask = mask * mask * (3.0 - 2.0 * mask)
    
    img = image_np.astype(np.float32)
    luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    highlight_protect = 1.0 - np.clip((luminance - 215.0) / 40.0, 0.0, 1.0)
    
    gain = 1.0 + 0.48 * vignette_mult * mask * highlight_protect
    img *= gain[:, :, None]
    
    if focal < 28.0 and is_raw:
        r_corr = 1.0 - 0.0012 * mask
        b_corr = 1.0 + 0.0012 * mask
        img[:, :, 0] *= r_corr
        img[:, :, 2] *= b_corr
        
    if is_raw:
        corner = mask * highlight_protect
        img[:, :, 2] *= (1.0 - 0.045 * corner)
        img[:, :, 0] *= (1.0 + 0.02 * corner)
    
    return np.clip(img, 0, 255).astype(np.uint8)

def optimize_whites_blacks_clipping(lf: np.ndarray, whites_pct: float = 0.08, blacks_pct: float = 0.08) -> np.ndarray:
    """
    Whites & Blacks Clipping Optimizer:
    Finds the exact percentiles for Whites and Blacks to maximize dynamic range
    without clipping details, and applies a soft range stretch.
    Capped to prevent crushing on synthetic/flat images.
    """
    p_black = float(np.percentile(lf, blacks_pct))
    p_white = float(np.percentile(lf, 100.0 - whites_pct))
    
    p_black = min(p_black, 0.025)
    p_white = max(p_white, 0.975)
    
    stretched = (lf - p_black) / max(p_white - p_black, 1e-5)
    return np.clip(stretched, 0.0, 1.0)

def apply_deep_denoising(image_np: np.ndarray, denoise_strength: float, metadata: dict) -> np.ndarray:
    """
    Sensor-Level Deep Denoising simulation:
    Separates chrominance (color noise) and luminance (luma noise).
    Applies strong edge-preserving bilateral filtering on chroma channels
    at high ISO speeds, and a gentle bilateral filter on luma.
    """
    if denoise_strength <= 0:
        return image_np.copy()
        
    iso = metadata.get('iso', 100.0)
    
    ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    if iso >= 800:
        d_chroma = int(5 + denoise_strength * 4)
        sigma_chroma = 15 + denoise_strength * 30
        cr_denoised = cv2.bilateralFilter(cr, d_chroma, sigma_chroma, sigma_chroma)
        cb_denoised = cv2.bilateralFilter(cb, d_chroma, sigma_chroma, sigma_chroma)
    else:
        d_chroma = 5
        sigma_chroma = 10 + denoise_strength * 10
        cr_denoised = cv2.bilateralFilter(cr, d_chroma, sigma_chroma, sigma_chroma)
        cb_denoised = cv2.bilateralFilter(cb, d_chroma, sigma_chroma, sigma_chroma)
        
    d_luma = int(3 + denoise_strength * 2)
    sigma_luma = 6 + denoise_strength * 12
    y_denoised = cv2.bilateralFilter(y, d_luma, sigma_luma, sigma_luma)
    
    ycrcb_denoised = cv2.merge((y_denoised, cr_denoised, cb_denoised))
    return cv2.cvtColor(ycrcb_denoised, cv2.COLOR_YCrCb2RGB)

def predict_sensei_sliders(analysis, metadata: dict) -> dict:
    """
    Sensei AI Joint Predictor for co-dependent image adjustments.
    Analyzes scene statistics and metadata (ISO, Lens, Aperture)
    to output Lightroom-style sliders.
    """
    iso = metadata.get('iso', 100.0)
    is_raw = metadata.get('is_raw', False)
    
    target_lum = 120.0 if analysis.has_skin else 115.0
    lum_diff = target_lum - analysis.mean_luminance
    max_exp_boost = 0.58 if is_raw else 0.35
    exposure = float(np.clip(lum_diff / 100.0, -0.30, max_exp_boost))
    
    if analysis.scene == "backlit_high_contrast":
        highlights = -0.65
        shadows = 0.55
        contrast = 0.05
        dehaze = 0.15
    elif analysis.scene == "outdoor_highlight":
        highlights = -0.75
        shadows = 0.30
        contrast = 0.18
        dehaze = 0.10
        if analysis.highlight_clip_ratio > 0.04:
            exposure = max(exposure - 0.15, -0.30)
    elif analysis.scene == "low_light":
        highlights = -0.20
        shadows = 0.60 if is_raw else 0.40
        contrast = 0.14
        dehaze = 0.08
        exposure = exposure + 0.12
    elif analysis.scene == "outdoor_balanced":
        highlights = -0.30
        shadows = 0.22
        contrast = 0.16
        dehaze = 0.05
    else:  # indoor_balanced
        highlights = -0.22
        shadows = 0.32
        contrast = 0.10
        dehaze = 0.03
        
    if analysis.has_subject:
        shadows = min(0.68, shadows + 0.10)
        
    if iso >= 6400:
        denoise = 0.60
    elif iso >= 3200:
        denoise = 0.45
    elif iso >= 1600:
        denoise = 0.30
    elif iso >= 800:
        denoise = 0.18
    else:
        denoise = 0.10 if analysis.noise_level > 15 else 0.0
        
    if is_raw and analysis.noise_level > 12:
        denoise = max(denoise, 0.25)
        
    vibrance = 0.12
    clarity = 0.08
    if analysis.has_skin:
        vibrance = 0.08
        clarity = 0.05
    elif analysis.scene == "outdoor_balanced":
        vibrance = 0.18
        clarity = 0.12
        
    whites = 0.08
    blacks = 0.08
    
    return {
        'exposure': exposure,
        'contrast': contrast,
        'highlights': highlights,
        'shadows': shadows,
        'whites': whites,
        'blacks': blacks,
        'dehaze': dehaze,
        'vibrance': vibrance,
        'clarity': clarity,
        'denoise': denoise
    }

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




_bilateral_cache = {}

def bilateral_grid_smooth(L: np.ndarray, s_space: int = 16, s_range: float = 0.1) -> np.ndarray:
    """
    Bilateral Grid Filter for edge-preserving smoothing.
    L: float32 numpy array of shape (H, W), values in [0, 1].
    """
    key = (id(L), L.shape, s_space, s_range, float(L.flat[0]), float(L.flat[-1]), float(L.flat[::2000].sum()) if L.size > 2000 else float(L.sum()))
    if key in _bilateral_cache:
        return _bilateral_cache[key]
        
    h_orig, w_orig = L.shape
    max_dim = 1024
    scale = 1.0
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        new_w = max(1, int(w_orig * scale))
        new_h = max(1, int(h_orig * scale))
        L_small = cv2.resize(L, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        L_small = L
        
    h, w = L_small.shape
    s_space = min(s_space, max(1, min(h, w) // 4))
    gh = int(np.ceil(h / s_space))
    gw = int(np.ceil(w / s_space))
    gz = int(np.ceil(1.0 / s_range)) + 1
    
    grid_val = np.zeros((gh, gw, gz), dtype=np.float32)
    grid_w = np.zeros((gh, gw, gz), dtype=np.float32)
    
    y_coords = np.arange(h, dtype=np.float32) / s_space
    x_coords = np.arange(w, dtype=np.float32) / s_space
    yy_coords, xx_coords = np.meshgrid(y_coords, x_coords, indexing='ij')
    zz_coords = L_small / s_range
    
    gy = np.clip(np.round(yy_coords).astype(np.int32), 0, gh - 1)
    gx = np.clip(np.round(xx_coords).astype(np.int32), 0, gw - 1)
    gz_idx = np.clip(np.round(zz_coords).astype(np.int32), 0, gz - 1)
    
    flat_idx = gy * (gw * gz) + gx * gz + gz_idx
    np.add.at(grid_val.ravel(), flat_idx.ravel(), L_small.ravel())
    np.add.at(grid_w.ravel(), flat_idx.ravel(), 1.0)
    
    blurred_val = np.zeros_like(grid_val)
    blurred_w = np.zeros_like(grid_w)
    for z in range(gz):
        if gh >= 3 and gw >= 3:
            blurred_val[:, :, z] = cv2.GaussianBlur(grid_val[:, :, z], (3, 3), 0)
            blurred_w[:, :, z] = cv2.GaussianBlur(grid_w[:, :, z], (3, 3), 0)
        else:
            blurred_val[:, :, z] = grid_val[:, :, z]
            blurred_w[:, :, z] = grid_w[:, :, z]
            
    final_val = np.zeros_like(blurred_val)
    final_w = np.zeros_like(blurred_w)
    for z in range(gz):
        z_prev = max(0, z - 1)
        z_next = min(gz - 1, z + 1)
        final_val[:, :, z] = 0.25 * blurred_val[:, :, z_prev] + 0.5 * blurred_val[:, :, z] + 0.25 * blurred_val[:, :, z_next]
        final_w[:, :, z] = 0.25 * blurred_w[:, :, z_prev] + 0.5 * blurred_w[:, :, z] + 0.25 * blurred_w[:, :, z_next]
        
    final_w = np.maximum(final_w, 1e-5)
    grid_normalized = final_val / final_w
    
    y0 = np.floor(yy_coords).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, gh - 1)
    y0 = np.clip(y0, 0, gh - 1)
    
    x0 = np.floor(xx_coords).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, gw - 1)
    x0 = np.clip(x0, 0, gw - 1)
    
    z0 = np.floor(zz_coords).astype(np.int32)
    z1 = np.clip(z0 + 1, 0, gz - 1)
    z0 = np.clip(z0, 0, gz - 1)
    
    yd = yy_coords - y0
    xd = xx_coords - x0
    zd = zz_coords - z0
    
    c000 = grid_normalized[y0, x0, z0]
    c001 = grid_normalized[y0, x0, z1]
    c010 = grid_normalized[y0, x1, z0]
    c011 = grid_normalized[y0, x1, z1]
    c100 = grid_normalized[y1, x0, z0]
    c101 = grid_normalized[y1, x0, z1]
    c110 = grid_normalized[y1, x1, z0]
    c111 = grid_normalized[y1, x1, z1]
    
    c00 = c000 * (1 - zd) + c001 * zd
    c01 = c010 * (1 - zd) + c011 * zd
    c10 = c100 * (1 - zd) + c101 * zd
    c11 = c110 * (1 - zd) + c111 * zd
    
    c0 = c00 * (1 - xd) + c01 * xd
    c1 = c10 * (1 - xd) + c11 * xd
    
    base_small = c0 * (1 - yd) + c1 * yd
    if scale != 1.0:
        base = cv2.resize(base_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    else:
        base = base_small
        
    if len(_bilateral_cache) > 30:
        _bilateral_cache.clear()
    _bilateral_cache[key] = base
    return base


def classify_wb_preset(image_np: np.ndarray, neutral_mask: np.ndarray = None) -> str:
    """
    Classify the light source type (Daylight/Tungsten/Shade/Fluorescent) based on
    color statistics of neutral regions or overall image.
    """
    if neutral_mask is not None and np.mean(neutral_mask) > 1.0:
        neutral_pts = image_np[neutral_mask > 128]
    else:
        # Fallback: find low saturation, mid-brightness pixels
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        neutral_pts = image_np[(s < 25) & (v > 40) & (v < 220)]
        
    if len(neutral_pts) < 100:
        mean_rgb = np.mean(image_np, axis=(0, 1))
    else:
        mean_rgb = np.mean(neutral_pts, axis=0)
        
    r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
    
    # Avoid division by zero
    r = max(r, 1.0)
    g = max(g, 1.0)
    b = max(b, 1.0)
    
    r_b_ratio = r / b
    g_rb_ratio = g / (0.5 * (r + b))
    
    if r_b_ratio > 1.35:
        return "Tungsten"
    elif r_b_ratio < 0.85:
        return "Shade"
    elif g_rb_ratio > 1.06:
        return "Fluorescent"
    else:
        return "Daylight"


def apply_wb_preset_correction(image_np: np.ndarray, preset: str) -> np.ndarray:
    """
    Apply temperature/tint correction gains to R, G, B channels based on classified preset.
    """
    if preset == "Daylight":
        return image_np.copy()
        
    img_float = image_np.astype(np.float32)
    if preset == "Tungsten":
        # Cool down warm orange light
        img_float[:, :, 0] *= 0.88  # Red
        img_float[:, :, 1] *= 0.98  # Green
        img_float[:, :, 2] *= 1.12  # Blue
    elif preset == "Shade":
        # Warm up cool blueish light
        img_float[:, :, 0] *= 1.12  # Red
        img_float[:, :, 1] *= 1.02  # Green
        img_float[:, :, 2] *= 0.88  # Blue
    elif preset == "Fluorescent":
        # Shift green cast to magenta
        img_float[:, :, 0] *= 1.02  # Red
        img_float[:, :, 1] *= 0.94  # Green
        img_float[:, :, 2] *= 1.04  # Blue
        
    return np.clip(img_float, 0, 255).astype(np.uint8)


def apply_hdr_fusion(image_np: np.ndarray, masks) -> np.ndarray:
    """
    Single-Image HDR Fusion:
    Generates under-exposed and over-exposed virtual exposures,
    then blends them using shadow and highlight masks to expand dynamic range.
    """
    img_float = image_np.astype(np.float32)
    under = img_float * (2.0 ** -1.2)
    over = img_float * (2.0 ** 1.2)
    
    shadow_mask = (masks.shadow_noise.astype(np.float32) / 255.0)[:, :, None]
    high_mask = ((masks.lamp_highlight + masks.sky_bright_outdoor) / 2.0).astype(np.float32) / 255.0
    high_mask = np.clip(high_mask, 0.0, 1.0)[:, :, None]
    
    hdr = img_float * (1.0 - shadow_mask - high_mask) + over * shadow_mask + under * high_mask
    return np.clip(hdr, 0, 255).astype(np.uint8)

def apply_hsl_wheel(image_np: np.ndarray, masks) -> np.ndarray:
    """
    Advanced HSL Wheel:
    Adjusts specific color channels (Greens, Blues, and Skin Tones/Orange)
    to enrich landscape leaves, skies, and keep skin natural.
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    green_mask = (h >= 35) & (h <= 85)
    blue_mask = (h >= 95) & (h <= 135)
    warm_mask = (h >= 5) & (h <= 25)
    
    s[green_mask] = np.clip(s[green_mask] * 1.22, 0, 255)
    h[green_mask] = np.clip(h[green_mask] + 2.0, 0, 180)
    v[green_mask] = np.clip(v[green_mask] * 0.92, 0, 255)
    
    s[blue_mask] = np.clip(s[blue_mask] * 1.30, 0, 255)
    v[blue_mask] = np.clip(v[blue_mask] * 0.88, 0, 255)
    
    skin_pixels = warm_mask & (masks.skin > 128)
    s[skin_pixels] = np.clip(s[skin_pixels] * 0.98, 0, 255)
    v[skin_pixels] = np.clip(v[skin_pixels] * 1.05, 0, 255)
    
    hsv_new = cv2.merge((h, s, v))
    return cv2.cvtColor(np.clip(hsv_new, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

def apply_subject_bokeh(image_np: np.ndarray, masks) -> np.ndarray:
    """
    AI Subject Bokeh:
    Separates the subject from the background using face_person mask,
    and applies a smooth bokeh/background blur.
    """
    bg_mask = (masks.background.astype(np.float32) / 255.0)[:, :, None]
    face_mask = (masks.face_person.astype(np.float32) / 255.0)[:, :, None]
    
    blur = cv2.GaussianBlur(image_np, (15, 15), 0)
    blend_weight = bg_mask * (1.0 - face_mask * 0.85)
    blend_weight = np.clip(blend_weight, 0.0, 1.0)
    
    bokeh = image_np.astype(np.float32) * (1.0 - blend_weight) + blur.astype(np.float32) * blend_weight
    return np.clip(bokeh, 0, 255).astype(np.uint8)

def apply_cpl_simulation(image_np: np.ndarray, masks) -> np.ndarray:
    """
    Polarizing Filter (CPL) Simulation:
    Darkens and polarizes the sky region with a vertical gradient,
    increasing blue saturation and reducing glare.
    """
    sky_mask = (masks.sky_bright_outdoor.astype(np.float32) / 255.0)
    if np.mean(sky_mask) < 0.01:
        return image_np.copy()
        
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    
    height, width = image_np.shape[:2]
    yy, _ = np.mgrid[0:height, 0:width]
    grad = 1.0 - (yy / height)
    cpl_mask = sky_mask * grad
    
    sky_pixels = (h >= 95) & (h <= 135) & (cpl_mask > 0.1)
    
    s[sky_pixels] = np.clip(s[sky_pixels] * (1.0 + 0.35 * cpl_mask[sky_pixels]), 0, 255)
    v[sky_pixels] = np.clip(v[sky_pixels] * (1.0 - 0.20 * cpl_mask[sky_pixels]), 0, 255)
    h[sky_pixels] = np.clip(h[sky_pixels] - 2.0 * cpl_mask[sky_pixels], 0, 180)
    
    hsv_new = cv2.merge((h, s, v))
    return cv2.cvtColor(np.clip(hsv_new, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def _skin_mask_rgb(image_np: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    return cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32) / 255.0


def _safe_gray_world_white_balance(image_np: np.ndarray, strength: float, neutral_wall_mask: np.ndarray = None) -> np.ndarray:
    if strength <= 0:
        return image_np
    rgb = image_np.astype(np.float32)
    use_guided = False
    if neutral_wall_mask is not None:
        neutral = neutral_wall_mask > 128
        if int(np.count_nonzero(neutral)) > image_np.shape[0] * image_np.shape[1] * 0.005:
            use_guided = True
    if not use_guided:
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
    smart_strength: str = "Event Balanced",
    masks = None,
    metadata: dict = None
) -> np.ndarray:
    """Apply scene-aware Smart Auto adjustments from image analysis, refined with region masks."""
    analysis = analyze_image(image_np, is_raw=is_raw)
    
    if metadata is None:
        metadata = {'focal_length': 35.0, 'aperture': 5.6, 'iso': 100.0, 'is_raw': is_raw, 'lens_model': 'Unknown'}
        
    sliders = predict_sensei_sliders(analysis, metadata)
    
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
    strength_key = (smart_strength or "Event Balanced").strip().lower()
    strength = {
        "natural": {"exposure": 0.86, "shadow": 0.78, "contrast": 0.72, "vibrance": 0.62, "clarity": 0.65},
        "event balanced": {"exposure": 1.0, "shadow": 1.0, "contrast": 1.0, "vibrance": 1.0, "clarity": 1.0},
        "event bright": {"exposure": 1.12, "shadow": 1.12, "contrast": 0.95, "vibrance": 1.05, "clarity": 0.95},
        "strong pop": {"exposure": 1.08, "shadow": 1.05, "contrast": 1.20, "vibrance": 1.28, "clarity": 1.12},
    }.get(strength_key, {"exposure": 1.0, "shadow": 1.0, "contrast": 1.0, "vibrance": 1.0, "clarity": 1.0})

    is_final_render = (masks is None)
    if masks is None:
        from app.region_detection import detect_regions
        masks = detect_regions(image_np, is_raw=is_raw)

    # 1. Single-Image HDR Fusion (Apply at the start of adjustments to extend dynamic range)
    if is_final_render:
        image_np = apply_hdr_fusion(image_np, masks)

    skin_mask_float = masks.skin.astype(np.float32) / 255.0
    face_person_float = masks.face_person.astype(np.float32) / 255.0
    lamp_highlight_float = masks.lamp_highlight.astype(np.float32) / 255.0
    sky_bright_outdoor_float = masks.sky_bright_outdoor.astype(np.float32) / 255.0
    background_float = masks.background.astype(np.float32) / 255.0
    shadow_noise_float = masks.shadow_noise.astype(np.float32) / 255.0
    neutral_wall_floor_float = masks.neutral_wall_floor.astype(np.float32) / 255.0
    
    # Suppress false positive sky mask on walls/ceilings for indoor scenes
    if analysis.scene in {"indoor_balanced", "low_light"}:
        sky_bright_outdoor_float = np.zeros_like(sky_bright_outdoor_float)
        
    # Restrict skin mask to the proximity of detected faces/people to prevent wall false positives
    skin_mask_float = skin_mask_float * np.clip(face_person_float * 1.8, 0.0, 1.0)
    
    # Suppress false positive lamp highlight mask for outdoor scenes to prevent grayscale bleeding on shirts/skin/hair
    if analysis.scene in {"outdoor_highlight", "outdoor_balanced"}:
        lamp_highlight_float = np.zeros_like(lamp_highlight_float)

    if is_raw and recommend_adjustments(analysis).correct_vignette:
        image_np = correct_raw_corner_shading(image_np, strength=0.82, metadata=metadata)

    is_people_indoor = analysis.has_skin and analysis.has_subject and analysis.scene in {"indoor_balanced", "low_light", "backlit_high_contrast"}

    wb_strength = (0.22 if is_people_indoor else 0.12) * profile.get("wb_mult", 1.0)
    image_np = _safe_gray_world_white_balance(image_np, wb_strength, neutral_wall_mask=masks.neutral_wall_floor)

    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    lf = np.clip(l.astype(np.float32) / 255.0, 0.0, 1.0)
    lf_orig = lf.copy()

    lf = optimize_whites_blacks_clipping(lf, whites_pct=sliders['whites'], blacks_pct=sliders['blacks'])

    base = bilateral_grid_smooth(lf, s_space=16, s_range=0.1)
    detail = lf - base

    dehaze_amount = sliders.get('dehaze', 0.0)
    if dehaze_amount > 0:
        local_mean = cv2.boxFilter(base, -1, (31, 31))
        dehaze_mask = np.clip(1.0 - skin_mask_float - 0.5 * face_person_float, 0.0, 1.0)
        base = base + dehaze_amount * (base - local_mean) * dehaze_mask
        base = np.clip(base, 0.0, 1.0)

    h, w = lf.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    sigma_x, sigma_y = max(w * 0.34, 1.0), max(h * 0.34, 1.0)
    center_weight = np.exp(-(((xx - cx) ** 2) / (2 * sigma_x ** 2) + ((yy - cy) ** 2) / (2 * sigma_y ** 2))).astype(np.float32)
    
    subject_weight = np.clip(0.40 * center_weight + 0.50 * skin_mask_float + 0.65 * face_person_float, 0.0, 1.0)
    
    highlight_protect = 1.0 - np.clip((base - 0.78) / 0.22, 0.0, 1.0) ** 2
    highlight_protect = highlight_protect * (1.0 - lamp_highlight_float)

    exposure = (sliders['exposure'] + profile.get("exposure_bias", 0.0)) * strength["exposure"]
    if is_people_indoor:
        exposure = max(exposure, 0.08 + profile.get("subject_boost", 0.0))
    if exposure:
        gain = 2.0 ** exposure
        base = base * (1.0 + (gain - 1.0) * highlight_protect * (0.55 + 0.45 * subject_weight))

    shadow_curve = base * np.square(1.0 - base)
    shadow_strength = sliders['shadows'] * (0.55 if is_people_indoor else 0.75) * profile.get("shadow_mult", 1.0) * strength["shadow"]
    base = base + shadow_strength * shadow_curve * (0.65 + 0.55 * subject_weight)

    if sliders['highlights'] < 0:
        shoulder = np.power(base, 2.4) * (1.0 - base)
        sky_pull_factor = 1.0 + 0.85 * sky_bright_outdoor_float
        base = base + sliders['highlights'] * profile.get("highlight_mult", 1.0) * 0.38 * shoulder * sky_pull_factor

    base = np.clip(base, 0.0, 1.0)
    contrast = sliders['contrast'] * (0.55 if is_people_indoor else 0.85) * profile.get("contrast_mult", 1.0) * strength["contrast"]
    if contrast:
        base = base + contrast * 0.20 * (base - 0.5) * (1.0 - np.abs(2.0 * base - 1.0))
        base = np.clip(base, 0.0, 1.0)

    M = cv2.moments(masks.face_person)
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cY, cX = h // 2, w // 2
        
    relight_radius = max(w, h) * 0.35
    dist = np.sqrt((xx - cX) ** 2 + (yy - cY) ** 2)
    radial_falloff = np.clip(1.0 - dist / relight_radius, 0.0, 1.0)
    radial_falloff = radial_falloff * radial_falloff * (3.0 - 2.0 * radial_falloff)
    
    relight_mask = radial_falloff * (0.3 + 0.7 * face_person_float)
    relight_boost = 0.065
    base = base + relight_boost * relight_mask * (1.0 - base)
    base = np.clip(base, 0.0, 1.0)

    lf = np.clip(base + detail, 0.0, 1.0)

    if sliders['highlights'] < 0:
        blur_orig = cv2.GaussianBlur(lf_orig, (5, 5), 0)
        texture = lf_orig - blur_orig
        bright_mask = np.clip((lf_orig - 0.65) / 0.35, 0.0, 1.0)
        recovery_mask = bright_mask * (1.0 + 0.6 * sky_bright_outdoor_float + 0.4 * lamp_highlight_float)
        recovery_mask = np.clip(recovery_mask, 0.0, 1.0)
        lf = np.clip(lf + texture * 1.6 * recovery_mask, 0.0, 1.0)

    l_new = np.clip(lf * 255.0, 0, 255).astype(np.uint8)
    ratio = np.clip((l_new.astype(np.float32) + 12.0) / (l.astype(np.float32) + 12.0), 0.82, 1.14)
    if is_people_indoor:
        ratio = 1.0 + (ratio - 1.0) * 0.35
    a_new = np.clip((a.astype(np.float32) - 128.0) * ratio + 128.0, 0, 255).astype(np.uint8)
    b_new = np.clip((b.astype(np.float32) - 128.0) * ratio + 128.0, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.merge((l_new, a_new, b_new)), cv2.COLOR_LAB2RGB)
    rgb = _protect_lamp_highlights(rgb)

    if sliders['vibrance'] > 0:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        s_ch = hsv[:, :, 1]
        vib = sliders['vibrance'] * (0.55 if is_people_indoor else 0.85) * profile.get("vibrance_mult", 1.0) * strength["vibrance"]
        boosted = s_ch + (255.0 - s_ch) * vib * (1.0 - s_ch / 255.0)
        
        sky_boost = sky_bright_outdoor_float * (1.0 - skin_mask_float) * 0.25 * (255.0 - s_ch)
        sky_boost = sky_boost * (1.0 - lamp_highlight_float)
        boosted = boosted + sky_boost
        
        s_ch = boosted * (1.0 - 0.82 * skin_mask_float) + s_ch * (0.82 * skin_mask_float)
        hsv[:, :, 1] = np.clip(s_ch, 0, 255)
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    if sliders['denoise'] > 0:
        denoised_rgb = apply_deep_denoising(rgb, sliders['denoise'], metadata)
        sn_weight = np.clip(0.15 + 0.85 * shadow_noise_float, 0.0, 1.0)
        rgb = (denoised_rgb.astype(np.float32) * sn_weight[:, :, None] + rgb.astype(np.float32) * (1.0 - sn_weight[:, :, None])).astype(np.uint8)

    if sliders['clarity'] > 0:
        blur = cv2.GaussianBlur(rgb, (0, 0), 1.0)
        amount = sliders['clarity'] * (0.22 if is_people_indoor else 0.36) * profile.get("clarity_mult", 1.0) * strength["clarity"]
        clarity_img = np.clip(rgb.astype(np.float32) + amount * (rgb.astype(np.float32) - blur.astype(np.float32)), 0, 255)
        
        smooth_skin = cv2.bilateralFilter(rgb, 5, 12, 12)
        rgb = (clarity_img * (1.0 - skin_mask_float[:, :, None]) + 
               smooth_skin.astype(np.float32) * skin_mask_float[:, :, None]).astype(np.uint8)

    if is_people_indoor:
        lab_skin = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_target = np.maximum(lab_skin[:, :, 0], l.astype(np.float32) + 5.0)
        skin_strength = np.clip(skin_mask_float * 0.55, 0.0, 0.55)
        lab_skin[:, :, 0] = lab_skin[:, :, 0] * (1.0 - skin_strength) + l_target * skin_strength
        rgb = cv2.cvtColor(np.clip(lab_skin, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    lamp_mask_float = masks.lamp_highlight.astype(np.float32) / 255.0
    gray_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[:, :, None]
    rgb = (rgb.astype(np.float32) * (1.0 - lamp_mask_float[:, :, None]) + 
           gray_img.astype(np.float32) * lamp_mask_float[:, :, None]).astype(np.uint8)

    rgb = _protect_lamp_highlights(rgb)
    return np.clip(rgb, 0, 255).astype(np.uint8)



def _lab_luminance(image_np: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)


def _center_luminance_mean(image_np: np.ndarray) -> float:
    l_channel = _lab_luminance(image_np)
    h, w = l_channel.shape
    crop = l_channel[int(h * 0.25):int(h * 0.75), int(w * 0.2):int(w * 0.7)]
    return float(np.mean(crop))


def _skin_luminance_mean(image_np: np.ndarray) -> float:
    skin = _skin_mask_rgb(image_np) > 0.35
    if int(np.count_nonzero(skin)) == 0:
        return 0.0
    l_channel = _lab_luminance(image_np)
    return float(np.mean(l_channel[skin]))


def _candidate_quality_score(original: np.ndarray, candidate: np.ndarray, is_raw: bool, profile: str, batch_context: dict | None = None, smart_strength: str = "Event Balanced") -> float:
    original_analysis = analyze_image(original, is_raw=is_raw)
    candidate_analysis = analyze_image(candidate, is_raw=is_raw)
    score = 100.0

    original_center = _center_luminance_mean(original)
    candidate_center = _center_luminance_mean(candidate)
    candidate_skin_l = _skin_luminance_mean(candidate)
    target_center = 143.0 if original_analysis.has_skin else 132.0
    target_skin = 146.0
    if batch_context:
        if batch_context.get("target_center"):
            target_center = 0.65 * target_center + 0.35 * float(batch_context["target_center"])
        if batch_context.get("target_skin"):
            target_skin = 0.70 * target_skin + 0.30 * float(batch_context["target_skin"])
    if (smart_strength or "").lower() == "event bright":
        target_center += 4.0
        target_skin += 3.0
    elif (smart_strength or "").lower() == "strong pop":
        target_center += 2.0

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
        return image_np.copy()
    scale = max_side / largest
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


def _score_smart_auto_profiles(image_np: np.ndarray, is_raw: bool = False, batch_context: dict | None = None, smart_strength: str = "Event Balanced") -> tuple[str, float, list[tuple[str, float]], np.ndarray]:
    scoring_source = _resize_for_scoring(image_np)
    
    # Detect masks once on the small scoring preview to keep scoring fast!
    from app.region_detection import detect_regions
    masks = detect_regions(scoring_source, is_raw=is_raw)
    
    profiles = _profiles_for_analysis(analyze_image(scoring_source, is_raw=is_raw))
    if _center_luminance_mean(scoring_source) > 150.0:
        profiles = [profile for profile in profiles if profile in {"natural_safe", "skin_safe", "balanced_pop", "highlight_safe"}] or ["natural_safe"]

    scored: list[tuple[str, float]] = []
    best_profile = "natural_safe"
    best_score = -1e9
    for profile in profiles:
        preview_candidate = apply_recommended_adjustments(scoring_source, is_raw=is_raw, candidate_profile=profile, smart_strength=smart_strength, masks=masks)
        score = _candidate_quality_score(scoring_source, preview_candidate, is_raw, profile, batch_context=batch_context, smart_strength=smart_strength)
        scored.append((profile, score))
        if score > best_score:
            best_profile = profile
            best_score = score
    if best_score < 20.0 and _center_luminance_mean(scoring_source) > 145.0:
        best_profile = "original_safe"
    return best_profile, best_score, sorted(scored, key=lambda item: item[1], reverse=True), scoring_source


def select_best_smart_auto_candidate(image_np: np.ndarray, is_raw: bool = False, batch_context: dict | None = None, smart_strength: str = "Event Balanced") -> tuple[np.ndarray, str, float]:
    preset = classify_wb_preset(image_np)
    corrected_img = apply_wb_preset_correction(image_np, preset)
    best_profile, best_score, _, _ = _score_smart_auto_profiles(corrected_img, is_raw=is_raw, batch_context=batch_context, smart_strength=smart_strength)
    if best_profile == "original_safe":
        return corrected_img.copy(), best_profile, best_score
    best_image = apply_recommended_adjustments(corrected_img, is_raw=is_raw, candidate_profile=best_profile, smart_strength=smart_strength)
    return best_image, best_profile, best_score


def get_smart_auto_profile_name(image_np: np.ndarray, is_raw: bool = False, batch_context: dict | None = None, smart_strength: str = "Event Balanced") -> str:
    """Return the Smart Auto v4 profile selected for this image."""
    preset = classify_wb_preset(image_np)
    corrected_img = apply_wb_preset_correction(image_np, preset)
    profile, _, _, _ = _score_smart_auto_profiles(corrected_img, is_raw=is_raw, batch_context=batch_context, smart_strength=smart_strength)
    return profile


def get_smart_auto_decision(image_np: np.ndarray, is_raw: bool = False, batch_context: dict | None = None, smart_strength: str = "Event Balanced") -> dict:
    """Return selected Smart Auto profile, score, and candidate ranking."""
    preset = classify_wb_preset(image_np)
    corrected_img = apply_wb_preset_correction(image_np, preset)
    profile, score, scored, _ = _score_smart_auto_profiles(corrected_img, is_raw=is_raw, batch_context=batch_context, smart_strength=smart_strength)
    return {
        "profile": profile,
        "score": float(score),
        "candidates": [(name, float(value)) for name, value in scored],
    }


def save_smart_auto_contact_sheet(image_np: np.ndarray, output_path: str, is_raw: bool = False, batch_context: dict | None = None, smart_strength: str = "Event Balanced") -> str:
    """Save a JPEG contact sheet with original and Smart Auto v4 candidates."""
    from PIL import Image, ImageDraw
    import os

    preset = classify_wb_preset(image_np)
    corrected_img = apply_wb_preset_correction(image_np, preset)

    selected, _, scored, scoring_source = _score_smart_auto_profiles(corrected_img, is_raw=is_raw, batch_context=batch_context, smart_strength=smart_strength)
    from app.region_detection import detect_regions
    masks = detect_regions(scoring_source, is_raw=is_raw)
    
    tiles: list[tuple[str, np.ndarray]] = [("original", scoring_source)]
    for profile, _ in scored:
        tiles.append((profile, apply_recommended_adjustments(scoring_source, is_raw=is_raw, candidate_profile=profile, smart_strength=smart_strength, masks=masks)))
    if selected == "original_safe":
        tiles.append(("selected: original_safe", scoring_source.copy()))

    tile_w = 360
    tile_h = 240
    label_h = 34
    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    score_lookup = {name: score for name, score in scored}

    for index, (name, arr) in enumerate(tiles):
        row, col = divmod(index, cols)
        pil = Image.fromarray(arr).convert("RGB")
        pil.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        x = col * tile_w + (tile_w - pil.width) // 2
        y = row * (tile_h + label_h) + label_h + (tile_h - pil.height) // 2
        sheet.paste(pil, (x, y))
        label = name
        if name in score_lookup:
            label += f" ({score_lookup[name]:.1f})"
        if name == selected or (selected == "original_safe" and name.startswith("selected")):
            label = "BEST: " + label
        draw.text((col * tile_w + 8, row * (tile_h + label_h) + 8), label, fill=(255, 255, 255))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sheet.save(output_path, "JPEG", quality=90)
    return output_path


def correct_raw_corner_shading(image_np: np.ndarray, strength: float = 0.95, metadata: dict = None) -> np.ndarray:
    """Correct RAW lens corner shading/vignetting with a soft radial gain."""
    if metadata is None:
        metadata = {'focal_length': 18.0, 'aperture': 4.5, 'is_raw': True, 'lens_model': 'Unknown'}
    return correct_lens_artifacts(image_np, metadata)


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

def smart_auto_enhance(image_np: np.ndarray, is_raw: bool = False, batch_context: dict | None = None, smart_strength: str = "Event Balanced") -> np.ndarray:
    """Choose the best Smart Auto candidate for this specific image."""
    best_image, _, _ = select_best_smart_auto_candidate(image_np, is_raw=is_raw, batch_context=batch_context, smart_strength=smart_strength)
    return best_image

def apply_autocorrect(image_np: np.ndarray, mode: str, is_raw: bool = False, batch_context: dict | None = None, smart_strength: str = "Event Balanced") -> np.ndarray:
    """
    Applies the chosen autocorrect mode to the image.
    """
    mode_clean = mode.strip().lower()
    
    if mode_clean == "off" or not mode_clean:
        return image_np

    preset = classify_wb_preset(image_np)
    img = apply_wb_preset_correction(image_np, preset)

    # Default to smart auto if Smart Auto is selected
    if "smart" in mode_clean or "auto" in mode_clean:
        return smart_auto_enhance(img, is_raw=is_raw, batch_context=batch_context, smart_strength=smart_strength)

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
