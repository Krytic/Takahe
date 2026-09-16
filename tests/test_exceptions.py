import pytest

import takahe


def test_TakaheFatalError_is_an_exception():
    with pytest.raises(takahe.exceptions.TakaheFatalError):
        raise takahe.exceptions.TakaheFatalError("test")


def test_TakaheTypeError_is_a_TakaheFatalError():
    assert issubclass(takahe.exceptions.TakaheTypeError,
                      takahe.exceptions.TakaheFatalError)

    with pytest.raises(takahe.exceptions.TakaheFatalError):
        raise takahe.exceptions.TakaheTypeError("test")


def test_TakaheWarning_is_a_Warning():
    assert issubclass(takahe.exceptions.TakaheWarning, Warning)


def test_TakaheUserWarning_is_both_UserWarning_and_TakaheWarning():
    assert issubclass(takahe.exceptions.TakaheUserWarning, UserWarning)
    assert issubclass(takahe.exceptions.TakaheUserWarning,
                      takahe.exceptions.TakaheWarning)

    with pytest.warns(takahe.exceptions.TakaheWarning):
        import warnings
        warnings.warn("test", takahe.exceptions.TakaheUserWarning)


def test_TakaheDeprecationWarning_is_both_DeprecationWarning_and_TakaheWarning():
    assert issubclass(takahe.exceptions.TakaheDeprecationWarning,
                      DeprecationWarning)
    assert issubclass(takahe.exceptions.TakaheDeprecationWarning,
                      takahe.exceptions.TakaheWarning)
