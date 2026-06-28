import numpy as np

from app.image_analysis import analyze_image, classify_scene, recommend_adjustments


def solid_rgb(value, size=(64, 64)):
    return np.full((size[1], size[0], 3), value, dtype=np.uint8)


def test_analyze_detects_overexposed_image():
    image = solid_rgb(252)
    analysis = analyze_image(image, is_raw=True)
    assert analysis.highlight_clip_ratio > 0.9
    assert analysis.exposure == "overexposed"
    assert analysis.scene == "outdoor_highlight"


def test_analyze_detects_underexposed_image():
    image = solid_rgb(25)
    analysis = analyze_image(image, is_raw=True)
    assert analysis.shadow_ratio > 0.9
    assert analysis.exposure == "underexposed"
    assert analysis.scene == "low_light"


def test_classify_scene_indoor_normal():
    assert classify_scene(mean_luminance=118, highlight_clip_ratio=0.01, shadow_ratio=0.2, saturation_mean=55, is_raw=False) == "indoor_balanced"


def test_recommend_adjustments_for_outdoor_highlight_avoids_exposure_boost():
    image = solid_rgb(240)
    analysis = analyze_image(image, is_raw=True)
    rec = recommend_adjustments(analysis)
    assert rec.exposure <= 0
    assert rec.highlights < 0
    assert rec.vibrance <= 0.05


def test_recommend_adjustments_for_underexposed_lifts_shadows():
    image = solid_rgb(25)
    analysis = analyze_image(image, is_raw=True)
    rec = recommend_adjustments(analysis)
    assert rec.exposure > 0
    assert rec.shadows > 0
    assert rec.denoise > 0
import numpy as np

from app.autocorrect import correct_raw_corner_shading
from app.image_analysis import analyze_image


def test_corner_shading_correction_reduces_vignette_metric():
    img = np.full((200, 300, 3), 140, dtype=np.uint8)
    yy, xx = np.mgrid[0:200, 0:300]
    radius = np.sqrt((xx - 149.5) ** 2 + (yy - 99.5) ** 2)
    radius = radius / radius.max()
    darken = 1.0 - 0.45 * np.clip((radius - 0.35) / 0.65, 0, 1)
    vignetted = np.clip(img.astype(np.float32) * darken[:, :, None], 0, 255).astype(np.uint8)

    before = analyze_image(vignetted, is_raw=True).vignette_strength
    after = analyze_image(correct_raw_corner_shading(vignetted, strength=0.9), is_raw=True).vignette_strength

    assert after < before
import numpy as np

from app.image_analysis import analyze_image, recommend_adjustments


def test_analysis_reports_highlight_regions():
    img = np.full((100, 100, 3), 80, dtype=np.uint8)
    img[:20, :20] = 255
    analysis = analyze_image(img, is_raw=False)
    assert analysis.highlight_clip_ratio > 0.03
    assert "highlight" in analysis.reasoning.lower()


def test_analysis_detects_skin_ratio():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [188, 128, 92]
    analysis = analyze_image(img, is_raw=False)
    assert analysis.skin_ratio > 0.5
    assert analysis.has_skin


def test_recommendations_include_subject_protection_when_skin_present():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :] = [188, 128, 92]
    analysis = analyze_image(img, is_raw=False)
    rec = recommend_adjustments(analysis)
    assert rec.protect_skin
    assert rec.reasoning


def test_analysis_detects_person_like_center_subject():
    img = np.full((120, 120, 3), 230, dtype=np.uint8)
    img[30:100, 45:75] = [120, 80, 60]
    analysis = analyze_image(img, is_raw=False)
    assert analysis.subject_ratio > 0
    assert analysis.has_subject


def test_analysis_exposes_face_detection_fields():
    img = np.full((96, 96, 3), 128, dtype=np.uint8)
    analysis = analyze_image(img, is_raw=False)
    assert isinstance(analysis.face_count, int)
    assert analysis.face_ratio >= 0.0
    assert isinstance(analysis.has_face, bool)
