import numpy as np
import pytest
import cv2
from app.autocorrect import (
    classify_wb_preset,
    apply_wb_preset_correction,
    bilateral_grid_smooth,
    apply_autocorrect,
    select_best_smart_auto_candidate
)
from app.region_detection import detect_regions

def test_smart_wb_preset_classifier():
    # 1. Tungsten (Warm/Orange)
    img_tungsten = np.full((100, 100, 3), [220, 160, 90], dtype=np.uint8)
    preset_tungsten = classify_wb_preset(img_tungsten)
    assert preset_tungsten == "Tungsten"
    corrected_tungsten = apply_wb_preset_correction(img_tungsten, preset_tungsten)
    # Corrected should have lower Red and higher Blue
    assert corrected_tungsten[0, 0, 0] < img_tungsten[0, 0, 0]
    assert corrected_tungsten[0, 0, 2] > img_tungsten[0, 0, 2]

    # 2. Shade (Cool/Blue)
    img_shade = np.full((100, 100, 3), [110, 150, 210], dtype=np.uint8)
    preset_shade = classify_wb_preset(img_shade)
    assert preset_shade == "Shade"
    corrected_shade = apply_wb_preset_correction(img_shade, preset_shade)
    # Corrected should have higher Red and lower Blue
    assert corrected_shade[0, 0, 0] > img_shade[0, 0, 0]
    assert corrected_shade[0, 0, 2] < img_shade[0, 0, 2]

    # 3. Fluorescent (Green cast)
    img_fluorescent = np.full((100, 100, 3), [130, 200, 125], dtype=np.uint8)
    preset_fluorescent = classify_wb_preset(img_fluorescent)
    assert preset_fluorescent == "Fluorescent"
    corrected_fluorescent = apply_wb_preset_correction(img_fluorescent, preset_fluorescent)
    # Corrected should have lower Green and slightly higher Red/Blue
    assert corrected_fluorescent[0, 0, 1] < img_fluorescent[0, 0, 1]
    assert corrected_fluorescent[0, 0, 0] > img_fluorescent[0, 0, 0]

    # 4. Daylight (Balanced)
    img_daylight = np.full((100, 100, 3), [150, 150, 145], dtype=np.uint8)
    preset_daylight = classify_wb_preset(img_daylight)
    assert preset_daylight == "Daylight"
    corrected_daylight = apply_wb_preset_correction(img_daylight, preset_daylight)
    # Corrected daylight should be identical
    np.testing.assert_array_equal(corrected_daylight, img_daylight)

def test_local_tone_mapping_bilateral_grid():
    # Create an image with a clear step edge
    L = np.zeros((64, 64), dtype=np.float32)
    L[:, :32] = 0.2
    L[:, 32:] = 0.8
    
    # Run bilateral grid smooth
    base = bilateral_grid_smooth(L, s_space=8, s_range=0.1)
    
    # Edge-preserving filter: base layer should keep the edge sharp
    # Difference on the edge should remain high, not blurred out
    assert base[30, 10] < 0.3
    assert base[30, 50] > 0.7
    # Diff should be small in flat areas
    assert abs(base[30, 10] - 0.2) < 0.05
    assert abs(base[30, 50] - 0.8) < 0.05

def test_highlight_recovery_engine():
    # Create a bright image with high-frequency details (alternating grid)
    img = np.full((120, 120, 3), 245, dtype=np.uint8)
    for y in range(0, 120, 2):
        for x in range(0, 120, 2):
            img[y, x] = [255, 255, 255]
            
    # Apply autocorrect in high-light/low-light mode where highlight compression is applied
    # Let's use event balanced/bright to trigger highlight edits
    res = apply_autocorrect(img, "Smart Auto (Recommended)", is_raw=True)
    
    # Check that high frequency texture is preserved and not flat-gray
    # Calculate local std deviation/difference of result highlights
    gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    local_diff = np.abs(gray[1:, :] - gray[:-1, :])
    # Textured areas should have non-zero variance
    assert np.mean(local_diff) > 0.05

def test_portrait_face_relighting():
    # Create an image with a simulated face in the center
    img = np.full((160, 160, 3), 100, dtype=np.uint8)
    # Add a mock person/face
    img[60:100, 60:100] = [180, 130, 95]
    
    # Get best smart auto result
    best_image, profile, score = select_best_smart_auto_candidate(img, is_raw=False)
    
    # The face/person area (centered around (80, 80)) should be highlighted compared to corner background
    lab_res = cv2.cvtColor(best_image, cv2.COLOR_RGB2LAB)
    l_face = np.mean(lab_res[70:90, 70:90, 0])
    l_bg = np.mean(lab_res[10:30, 10:30, 0])
    
    # Face region should be lifted elegantly and be brighter than background
    assert l_face > l_bg


def test_lens_artifact_correction():
    from app.autocorrect import correct_lens_artifacts
    # Create a dummy image
    img = np.full((120, 160, 3), 180, dtype=np.uint8)
    
    # Metadata for wide angle lens with wide aperture
    metadata = {
        'focal_length': 15.0,
        'aperture': 1.8,
        'is_raw': True,
        'lens_model': 'Mock RF 15-30mm'
    }
    
    corrected = correct_lens_artifacts(img, metadata)
    
    # Corner pixels should have vignette correction and chromatic aberration shift
    # Corners should be modified
    assert not np.array_equal(corrected[0, 0], img[0, 0])
    # Center should remain virtually identical or less changed
    assert np.mean(np.abs(corrected[60, 80].astype(float) - img[60, 80])) < np.mean(np.abs(corrected[0, 0].astype(float) - img[0, 0]))

def test_whites_blacks_clipping_optimizer():
    from app.autocorrect import optimize_whites_blacks_clipping
    # Create low contrast L channel
    lf = np.full((50, 50), 0.5, dtype=np.float32)
    lf[0, 0] = 0.4
    lf[-1, -1] = 0.6
    
    stretched = optimize_whites_blacks_clipping(lf, whites_pct=0.08, blacks_pct=0.08)
    
    # Range should be stretched
    assert stretched.min() <= lf.min()
    assert stretched.max() >= lf.max()

def test_deep_denoising():
    from app.autocorrect import apply_deep_denoising
    # Create an image with noise
    np.random.seed(42)
    img = np.full((80, 80, 3), 128, dtype=np.uint8)
    noise = np.random.randint(-15, 15, size=img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    metadata = {
        'iso': 6400.0,
        'focal_length': 35.0,
        'aperture': 4.0,
        'is_raw': True
    }
    
    denoised = apply_deep_denoising(img, denoise_strength=0.5, metadata=metadata)
    
    # Denoised image should have lower standard deviation (be smoother)
    std_orig = np.std(img)
    std_denoised = np.std(denoised)
    assert std_denoised < std_orig

def test_dehaze_and_skin_smoothing():
    # Create an image with background (dehaze target) and skin region (smoothing target)
    img = np.full((120, 160, 3), 120, dtype=np.uint8)
    # Skin area
    img[40:80, 40:80] = [188, 128, 92]
    # Noisy background
    img[:30, :30] = [100, 110, 120]
    
    # Run best candidate selection
    res, profile, score = select_best_smart_auto_candidate(img, is_raw=False)
    assert res.shape == img.shape
    assert profile

def test_hdr_fusion():
    from app.autocorrect import apply_hdr_fusion
    from app.region_detection import detect_regions
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Add a mock shadow and highlight
    img[:10, :10] = 20
    img[90:, 90:] = 250
    masks = detect_regions(img)
    hdr = apply_hdr_fusion(img, masks)
    assert hdr.shape == img.shape
    # Fused HDR should pull highlights down and push shadows up
    assert hdr[0, 0, 0] > img[0, 0, 0]
    assert hdr[95, 95, 0] < img[95, 95, 0]

def test_hsl_wheel():
    from app.autocorrect import apply_hsl_wheel
    from app.region_detection import detect_regions
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Green patch
    img[:50, :] = [50, 180, 50]
    # Blue patch
    img[50:, :] = [50, 50, 200]
    masks = detect_regions(img)
    adjusted = apply_hsl_wheel(img, masks)
    
    # Check that colors have been saturated/modified
    assert not np.array_equal(adjusted, img)

def test_subject_bokeh():
    from app.autocorrect import apply_subject_bokeh
    from app.region_detection import detect_regions
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Mock person
    img[40:80, 40:80] = [188, 128, 92]
    # Noisy background
    np.random.seed(42)
    img[:20, :20] = np.random.randint(0, 255, size=(20, 20, 3), dtype=np.uint8)
    
    masks = detect_regions(img)
    bokeh = apply_subject_bokeh(img, masks)
    
    # Background (top-left) should be blurred (lower variance/standard deviation)
    orig_bg_std = np.std(img[:20, :20])
    bokeh_bg_std = np.std(bokeh[:20, :20])
    assert bokeh_bg_std < orig_bg_std

def test_cpl_simulation():
    from app.autocorrect import apply_cpl_simulation
    from app.region_detection import detect_regions
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Sky region (blue)
    img[:40, :] = [100, 150, 255]
    masks = detect_regions(img)
    # Make sure sky mask is non-zero
    masks.sky_bright_outdoor[:40, :] = 255
    
    cpl = apply_cpl_simulation(img, masks)
    
    # Sky region should be shifted in color and darkened
    assert not np.array_equal(cpl[:40, :], img[:40, :])
    assert np.mean(cpl[:40, :, 2]) < np.mean(img[:40, :, 2]) # darkened
