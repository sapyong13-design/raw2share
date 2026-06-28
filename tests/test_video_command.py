from app.video_converter import build_ffmpeg_command

def test_build_ffmpeg_command_presets():
    input_file = "test_input.mov"
    output_file = "test_output.mp4"
    
    # 1. WhatsApp Balanced 1080p
    cmd1 = build_ffmpeg_command(input_file, output_file, "WhatsApp Balanced 1080p", "Keep original", faststart=True)
    assert "-y" in cmd1
    assert "-i" in cmd1
    assert "test_input.mov" in cmd1
    assert "test_output.mp4" in cmd1
    assert "-crf" in cmd1
    assert cmd1[cmd1.index("-crf") + 1] == "23"
    assert cmd1[cmd1.index("-preset") + 1] == "medium"
    assert cmd1[cmd1.index("-b:a") + 1] == "160k"
    assert "scale='if(gt(iw,1920),1920,iw)':-2" in cmd1
    assert "+faststart" in cmd1
    
    # 2. High Quality 1080p
    cmd2 = build_ffmpeg_command(input_file, output_file, "High Quality 1080p", "30 fps", faststart=False)
    assert cmd2[cmd2.index("-crf") + 1] == "20"
    assert cmd2[cmd2.index("-preset") + 1] == "slow"
    assert cmd2[cmd2.index("-b:a") + 1] == "192k"
    assert "scale='if(gt(iw,1920),1920,iw)':-2" in cmd2
    assert "-r" in cmd2
    assert cmd2[cmd2.index("-r") + 1] == "30"
    # should NOT have faststart
    assert "+faststart" not in cmd2

    # 3. Small File 720p
    cmd3 = build_ffmpeg_command(input_file, output_file, "Small File 720p", "Keep original", faststart=True)
    assert cmd3[cmd3.index("-crf") + 1] == "26"
    assert cmd3[cmd3.index("-preset") + 1] == "medium"
    assert cmd3[cmd3.index("-b:a") + 1] == "128k"
    assert "scale='if(gt(iw,1280),1280,iw)':-2" in cmd3

    # 4. Original Resolution
    cmd4 = build_ffmpeg_command(input_file, output_file, "Original Resolution High Quality", "Keep original", faststart=True)
    assert cmd4[cmd4.index("-crf") + 1] == "20"
    assert cmd4[cmd4.index("-preset") + 1] == "slow"
    # scaling filter should not be present
    assert "-vf" not in cmd4
