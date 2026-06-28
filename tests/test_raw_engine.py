from pathlib import Path

from app.raw_engine import RawEngine, detect_raw_engines, select_raw_engine


def test_detect_raw_engines_prefers_explicit_paths(tmp_path):
    rawtherapee = tmp_path / "rawtherapee-cli.exe"
    darktable = tmp_path / "darktable-cli.exe"
    rawtherapee.write_text("", encoding="utf-8")
    darktable.write_text("", encoding="utf-8")

    detected = detect_raw_engines(str(rawtherapee), str(darktable))

    assert detected.rawtherapee == str(rawtherapee)
    assert detected.darktable == str(darktable)


def test_select_raw_engine_auto_uses_preview_when_safe():
    assert select_raw_engine(
        RawEngine.AUTO,
        has_safe_preview=True,
        rawtherapee_path=None,
        darktable_path=None,
    ) == RawEngine.CAMERA_PREVIEW


def test_select_raw_engine_auto_uses_external_before_rawpy_when_preview_unsafe():
    assert select_raw_engine(
        RawEngine.AUTO,
        has_safe_preview=False,
        rawtherapee_path="C:/Program Files/RawTherapee/rawtherapee-cli.exe",
        darktable_path="C:/Program Files/darktable/bin/darktable-cli.exe",
    ) == RawEngine.RAWTHERAPEE


def test_select_raw_engine_auto_falls_back_to_rawpy():
    assert select_raw_engine(
        RawEngine.AUTO,
        has_safe_preview=False,
        rawtherapee_path=None,
        darktable_path=None,
    ) == RawEngine.RAWPY

from app.raw_engine import build_rawtherapee_command, build_darktable_command


def test_build_rawtherapee_command():
    cmd = build_rawtherapee_command("rt.exe", "in.CR3", "out.jpg", quality=92)
    assert cmd[0] == "rt.exe"
    assert "-o" in cmd
    assert "out.jpg" in cmd
    assert "-j92" in cmd
    assert "in.CR3" in cmd


def test_build_darktable_command():
    cmd = build_darktable_command("dt.exe", "in.CR3", "out.jpg", quality=92)
    assert cmd[0] == "dt.exe"
    assert "in.CR3" in cmd
    assert "out.jpg" in cmd
    assert "--core" in cmd
    assert any("quality=92" in item for item in cmd)
