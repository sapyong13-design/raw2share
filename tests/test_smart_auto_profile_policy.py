from types import SimpleNamespace

from app.autocorrect import _profiles_for_analysis


def test_outdoor_balanced_profiles_exclude_people_bright_false_skin_profile():
    analysis = SimpleNamespace(scene="outdoor_balanced")

    profiles = _profiles_for_analysis(analysis)

    assert "people_bright" not in profiles
    assert "event_bright" not in profiles
