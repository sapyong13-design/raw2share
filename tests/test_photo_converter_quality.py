from PIL import Image, JpegImagePlugin

from app.photo_converter import convert_raw_to_jpg


def test_jpeg_conversion_preserves_full_chroma_quality(tmp_path):
    input_path = tmp_path / "color_detail_source.jpg"
    output_path = tmp_path / "color_detail.jpg"

    img = Image.new("RGB", (64, 64))
    pixels = img.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = (255, 32, 32) if (x + y) % 2 == 0 else (32, 64, 255)
    img.save(input_path, "JPEG", quality=100, subsampling=0)

    result = convert_raw_to_jpg(
        str(input_path),
        str(output_path),
        {
            "quality": 98,
            "keep_resolution": True,
            "autocorrect_mode": "Off",
            "copy_exif": False,
        },
    )

    assert result["success"], result["message"]
    with Image.open(output_path) as saved:
        assert JpegImagePlugin.get_sampling(saved) == 0
