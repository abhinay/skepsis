import skepsis


def test_version_is_string() -> None:
    assert isinstance(skepsis.__version__, str)
    assert skepsis.__version__.startswith("0.1.0")
