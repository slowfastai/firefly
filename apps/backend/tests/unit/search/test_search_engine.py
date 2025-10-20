from search.search_engine import remove_punctuation


def test_remove_punctuation_basic():
    assert remove_punctuation("Hello, world!") == "Hello world"


def test_remove_punctuation_mixed_chars():
    # Hyphen, underscore, dots are removed; letters/digits remain
    assert remove_punctuation("a-b_c v1.0.0-beta") == "abc v100beta"


def test_remove_punctuation_only_punct():
    # Removes both Unicode punctuation ('P') and symbols ('S')
    assert remove_punctuation("!@#$%^&*()[]{};:'\",.<>/?|-_=+") == ""


def test_remove_punctuation_empty():
    assert remove_punctuation("") == ""


def test_remove_punctuation_unicode_punctuation_not_removed():
    # CJK punctuation is removed by default (Unicode category 'P')
    assert remove_punctuation("你好，世界！") == "你好世界"


def test_remove_punctuation_keep_characters():
    # Using keep preserves specified punctuation
    assert remove_punctuation("你好，世界！", keep={"，", "！"}) == "你好，世界！"
