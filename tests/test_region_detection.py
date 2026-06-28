import numpy as np
import pytest
from app.region_detection import detect_regions, generate_region_debug_sheet
from app.photo_converter import convert_raw_to_jpg

def test_region_detection_shapes_and_types():
    # Create a dummy RGB image
    img = np.full((120, 160, 3), [170, 160, 140], dtype=np.uint8)
    # Add a mock skin region
    img[40:80, 50:100] = [180, 130, 95]
    # Add a mock highlight region
    img[10:30, 110:140] = [255, 255, 255]
    
    masks = detect_regions(img, is_raw=False)
    
    assert masks.face_person.shape == (120, 160)
    assert masks.skin.shape == (120, 160)
    assert masks.lamp_highlight.shape == (120, 160)
    assert masks.sky_bright_outdoor.shape == (120, 160)
    assert masks.background.shape == (120, 160)
    assert masks.shadow_noise.shape == (120, 160)
    assert masks.neutral_wall_floor.shape == (120, 160)
    
    assert masks.face_person.dtype == np.uint8
    assert masks.skin.dtype == np.uint8
    assert masks.lamp_highlight.dtype == np.uint8
    assert masks.sky_bright_outdoor.dtype == np.uint8
    assert masks.background.dtype == np.uint8
    assert masks.shadow_noise.dtype == np.uint8
    assert masks.neutral_wall_floor.dtype == np.uint8

def test_generate_region_debug_sheet():
    img = np.full((120, 160, 3), [170, 160, 140], dtype=np.uint8)
    masks = detect_regions(img, is_raw=False)
    sheet = generate_region_debug_sheet(img, masks)
    
    assert sheet.ndim == 3
    assert sheet.shape[2] == 3
    assert sheet.dtype == np.uint8

def test_photo_converter_saves_ai_debug(tmp_path):
    img_path = tmp_path / "test_input.jpg"
    out_path = tmp_path / "test_output.jpg"
    
    from PIL import Image
    # Create and save a small dummy image
    img = Image.new("RGB", (200, 200), (170, 160, 140))
    img.save(img_path)
    
    settings = {
        'quality': 90,
        'keep_resolution': True,
        'autocorrect_mode': 'Smart Auto (Recommended)',
        'use_camera_wb': True,
        'copy_exif': False,
        'save_contact_sheet': False,
        'save_ai_debug': True
    }
    
    res = convert_raw_to_jpg(str(img_path), str(out_path), settings)
    assert res['success']
    assert out_path.exists()
    
    debug_path = tmp_path / "test_output_ai_debug.jpg"
    assert debug_path.exists()
    assert debug_path.stat().st_size > 0
