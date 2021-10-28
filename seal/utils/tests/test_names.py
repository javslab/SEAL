from seal.utils.names import get_random_name

def test_format_get_random_name():
    test_name = get_random_name()
    assert "_" in test_name, "test_name does not contain an underscore"