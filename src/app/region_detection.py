import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class RegionMasks:
    face_person: np.ndarray      # uint8, 0-255
    skin: np.ndarray             # uint8, 0-255
    lamp_highlight: np.ndarray    # uint8, 0-255
    sky_bright_outdoor: np.ndarray# uint8, 0-255
    background: np.ndarray       # uint8, 0-255
    shadow_noise: np.ndarray     # uint8, 0-255
    neutral_wall_floor: np.ndarray# uint8, 0-255

_region_cache = {}

def detect_regions(image_np: np.ndarray, is_raw: bool = False) -> RegionMasks:
    """
    Detect different semantic regions in an RGB image.
    Performs detection on a downscaled version for speed, then upscales the masks
    back to the original image dimensions.
    """
    key = (id(image_np), image_np.shape, is_raw, int(image_np.flat[0]), int(image_np.flat[-1]), float(image_np.flat[::2000].sum()) if image_np.size > 2000 else float(image_np.sum()))
    if key in _region_cache:
        return _region_cache[key]
        
    h_orig, w_orig = image_np.shape[:2]
    
    # Downscale for fast processing
    max_dim = 1024
    scale = 1.0
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / max(h_orig, w_orig)
        new_w = max(1, int(w_orig * scale))
        new_h = max(1, int(h_orig * scale))
        img_small = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_small = image_np.copy()
        
    h, w = img_small.shape[:2]
    
    # 1. Color conversions
    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(img_small, cv2.COLOR_RGB2YCrCb)
    lab = cv2.cvtColor(img_small, cv2.COLOR_RGB2LAB)
    
    # Extract channels
    l_chan = lab[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]
    
    # 2. Skin Mask (YCrCb heuristic)
    # Cr range: 133 to 173, Cb range: 77 to 127
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # 3. Face & Person Mask
    face_mask = np.zeros((h, w), dtype=np.uint8)
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if not cascade.empty():
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            for (x, y, fw, fh) in faces:
                # Draw face
                cv2.rectangle(face_mask, (x, y), (x + fw, y + fh), 255, -1)
                # Draw body/shoulder extension
                bx = max(0, x - int(fw * 0.4))
                by = y + fh
                bw = min(w, x + fw + int(fw * 0.4))
                bh = min(h, y + fh + int(fh * 2.5))
                cv2.rectangle(face_mask, (bx, by), (bw, bh), 180, -1)
    except Exception:
        pass
        
    # If no faces/people detected, fallback to center region ellipse
    if np.mean(face_mask) < 1.0:
        cv2.ellipse(face_mask, (w // 2, h // 2), (int(w * 0.28), int(h * 0.38)), 0, 0, 360, 120, -1)
        
    # Combine face/person mask and smooth it
    face_person_mask = cv2.GaussianBlur(face_mask, (15, 15), 0)
    
    # 4. Lamp / Clipped Highlight Mask
    # We look for extremely bright pixels (L >= 240)
    lamp_mask = cv2.inRange(l_chan, 240, 255)
    # Dilate to cover the glow/halo around lamps
    kernel_size = max(5, int(min(w, h) * 0.025))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    lamp_highlight = cv2.dilate(lamp_mask, kernel, iterations=1)
    lamp_highlight = cv2.GaussianBlur(lamp_highlight, (kernel_size, kernel_size), 0)
    
    # 5. Sky / Bright Outdoor Mask
    # High luminance in the upper part of the image
    sky_mask = np.zeros((h, w), dtype=np.uint8)
    sky_y_limit = int(h * 0.48)
    # Filter upper region for high brightness and relatively low saturation or blueish tones
    upper_l = l_chan[:sky_y_limit, :]
    upper_s = s_chan[:sky_y_limit, :]
    
    # Sky condition: L >= 170 and S <= 100 (excluding highly saturated foreground elements)
    sky_pixels = (upper_l >= 165) & (upper_s < 110)
    sky_mask[:sky_y_limit, :][sky_pixels] = 255
    # Smooth the sky mask
    sky_bright_outdoor = cv2.GaussianBlur(sky_mask, (21, 21), 0)
    
    # 6. Background Mask
    # Complement of face_person and skin
    foreground = np.maximum(face_person_mask, skin_mask)
    background = cv2.bitwise_not(foreground)
    background = cv2.GaussianBlur(background, (11, 11), 0)
    
    # 7. Shadow / Noise Mask
    # Low luminance pixels
    shadow_mask = cv2.inRange(l_chan, 0, 65)
    shadow_noise = cv2.GaussianBlur(shadow_mask, (11, 11), 0)
    
    # 8. Neutral Wall / Floor Mask
    # Mid-brightness and very low saturation
    neutral_mask = (l_chan >= 60) & (l_chan <= 210) & (s_chan < 22)
    neutral_wall_floor = np.zeros((h, w), dtype=np.uint8)
    neutral_wall_floor[neutral_mask] = 255
    neutral_wall_floor = cv2.GaussianBlur(neutral_wall_floor, (15, 15), 0)
    
    # Helper to upscale masks to original size if needed
    def upscale_mask(mask: np.ndarray) -> np.ndarray:
        if scale != 1.0:
            return cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        return mask
        
    res = RegionMasks(
        face_person=upscale_mask(face_person_mask),
        skin=upscale_mask(skin_mask),
        lamp_highlight=upscale_mask(lamp_highlight),
        sky_bright_outdoor=upscale_mask(sky_bright_outdoor),
        background=upscale_mask(background),
        shadow_noise=upscale_mask(shadow_noise),
        neutral_wall_floor=upscale_mask(neutral_wall_floor)
    )
    if len(_region_cache) > 30:
        _region_cache.clear()
    _region_cache[key] = res
    return res

def generate_region_debug_sheet(image_np: np.ndarray, masks: RegionMasks) -> np.ndarray:
    """
    Generate a 3x3 contact grid showing:
    [Original]     [Face/Person]       [Skin]
    [Lamp/High]    [Sky/Bright Out]    [Background]
    [Shadow/Noise] [Neutral Wall]      [Combined Overlay]
    """
    from PIL import Image, ImageDraw
    
    h_orig, w_orig = image_np.shape[:2]
    tile_w = 400
    tile_h = int(tile_w * (h_orig / w_orig))
    
    # Downsample original image for the sheet
    orig_small = cv2.resize(image_np, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    
    def prep_mask_tile(mask: np.ndarray) -> np.ndarray:
        m_small = cv2.resize(mask, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(m_small, cv2.COLOR_GRAY2RGB)
        
    # Draw a combined color overlay on the original
    # Face/Person: Red, Skin: Green, Lamp: Yellow, Sky: Blue, Shadow: Purple, Neutral: Cyan
    overlay = orig_small.copy().astype(np.float32)
    
    # Get small versions of masks for the overlay
    fp_s = cv2.resize(masks.face_person, (tile_w, tile_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    sk_s = cv2.resize(masks.skin, (tile_w, tile_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    lp_s = cv2.resize(masks.lamp_highlight, (tile_w, tile_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    sky_s = cv2.resize(masks.sky_bright_outdoor, (tile_w, tile_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    sh_s = cv2.resize(masks.shadow_noise, (tile_w, tile_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    nt_s = cv2.resize(masks.neutral_wall_floor, (tile_w, tile_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    
    # Blend colors
    # Red for Face/Person
    overlay[:, :, 0] = overlay[:, :, 0] * (1.0 - 0.45 * fp_s) + 255.0 * (0.45 * fp_s)
    # Green for Skin
    overlay[:, :, 1] = overlay[:, :, 1] * (1.0 - 0.45 * sk_s) + 255.0 * (0.45 * sk_s)
    # Yellow for Lamp (Red + Green)
    overlay[:, :, 0] = overlay[:, :, 0] * (1.0 - 0.5 * lp_s) + 255.0 * (0.5 * lp_s)
    overlay[:, :, 1] = overlay[:, :, 1] * (1.0 - 0.5 * lp_s) + 255.0 * (0.5 * lp_s)
    # Blue for Sky
    overlay[:, :, 2] = overlay[:, :, 2] * (1.0 - 0.4 * sky_s) + 255.0 * (0.4 * sky_s)
    # Cyan for Neutral (Green + Blue)
    overlay[:, :, 1] = overlay[:, :, 1] * (1.0 - 0.3 * nt_s) + 255.0 * (0.3 * nt_s)
    overlay[:, :, 2] = overlay[:, :, 2] * (1.0 - 0.3 * nt_s) + 255.0 * (0.3 * nt_s)
    # Purple for Shadow (Red + Blue)
    overlay[:, :, 0] = overlay[:, :, 0] * (1.0 - 0.3 * sh_s) + 180.0 * (0.3 * sh_s)
    overlay[:, :, 2] = overlay[:, :, 2] * (1.0 - 0.3 * sh_s) + 180.0 * (0.3 * sh_s)
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    tiles = [
        ("Original", orig_small),
        ("Face / Person Mask", prep_mask_tile(masks.face_person)),
        ("Skin Mask", prep_mask_tile(masks.skin)),
        ("Lamp / Highlight Mask", prep_mask_tile(masks.lamp_highlight)),
        ("Sky / Bright Outdoor Mask", prep_mask_tile(masks.sky_bright_outdoor)),
        ("Background Mask", prep_mask_tile(masks.background)),
        ("Shadow / Noise Mask", prep_mask_tile(masks.shadow_noise)),
        ("Neutral Wall / Floor Mask", prep_mask_tile(masks.neutral_wall_floor)),
        ("Combined Color Overlay", overlay)
    ]
    
    label_h = 30
    cols = 3
    rows = 3
    grid_w = cols * tile_w
    grid_h = rows * (tile_h + label_h)
    
    sheet = Image.new("RGB", (grid_w, grid_h), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)
    
    for idx, (name, arr) in enumerate(tiles):
        r, c = divmod(idx, cols)
        pil = Image.fromarray(arr)
        x = c * tile_w
        y = r * (tile_h + label_h) + label_h
        sheet.paste(pil, (x, y))
        draw.text((x + 10, r * (tile_h + label_h) + 8), name, fill=(240, 240, 240))
        
    return np.array(sheet)
