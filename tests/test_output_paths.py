import os
from app.utils import get_unique_filepath

def test_unique_filepath(tmp_path):
    target = tmp_path / "photo.jpg"
    
    # Non-conflicting path
    unique_path1 = get_unique_filepath(str(target))
    assert unique_path1 == str(target)
    
    # Create file to cause conflict
    target.touch()
    
    # First conflict -> should append _1
    unique_path2 = get_unique_filepath(str(target))
    assert unique_path2 == str(tmp_path / "photo_1.jpg")
    
    # Create the _1 file to cause second conflict
    (tmp_path / "photo_1.jpg").touch()
    
    # Second conflict -> should append _2
    unique_path3 = get_unique_filepath(str(target))
    assert unique_path3 == str(tmp_path / "photo_2.jpg")
