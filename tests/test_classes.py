from src.classes import CLASS_NAMES


def test_class_names_count():
    assert len(CLASS_NAMES) == 8


def test_class_names_alphabetical():
    assert CLASS_NAMES == sorted(CLASS_NAMES)


def test_class_names_no_duplicates():
    assert len(set(CLASS_NAMES)) == len(CLASS_NAMES)


def test_class_names_match_config():
    # configs/config.yaml declares num_classes: 8
    assert len(CLASS_NAMES) == 8