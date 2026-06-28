import os
from app.file_scanner import scan_path, scan_paths, is_photo, is_video

def test_extensions():
    assert is_photo("image.CR3")
    assert is_photo("photo.cr2")
    assert is_photo("photo.jpg")
    
    assert is_video("movie.mp4")
    assert is_video("clip.MOV")
    assert is_video("stream.mts")
    assert not is_video("stream.mp3")

def test_scan_path(tmp_path):
    # Create temp files
    photo1 = tmp_path / "img1.CR3"
    photo1.touch()
    
    video1 = tmp_path / "vid1.mp4"
    video1.touch()
    
    # Text file (should be ignored)
    txt_file = tmp_path / "notes.txt"
    txt_file.touch()
    
    # Subfolder
    sub = tmp_path / "sub"
    sub.mkdir()
    photo2 = sub / "img2.cr2"
    photo2.touch()
    video2 = sub / "vid2.MOV"
    video2.touch()
    
    # Non-recursive scan
    photos, videos = scan_path(str(tmp_path), recursive=False)
    assert len(photos) == 1
    assert len(videos) == 1
    assert os.path.basename(photos[0]).lower() == "img1.cr3"
    assert os.path.basename(videos[0]).lower() == "vid1.mp4"
    
    # Recursive scan
    photos_rec, videos_rec = scan_path(str(tmp_path), recursive=True)
    assert len(photos_rec) == 2
    assert len(videos_rec) == 2

def test_scan_paths_ignores_output(tmp_path):
    img = tmp_path / "input.cr3"
    img.touch()
    
    # Output dir
    output_dir = tmp_path / "RAW2Share_Output"
    output_dir.mkdir()
    
    out_img = output_dir / "output.cr3"
    out_img.touch()
    
    photos, videos = scan_path(str(tmp_path), recursive=True, output_dir=str(output_dir))
    # Should only scan input.cr3, not the one in RAW2Share_Output
    assert len(photos) == 1
    assert os.path.basename(photos[0]) == "input.cr3"

