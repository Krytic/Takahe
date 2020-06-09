import takahe
import pytest

def test_load_from_file():
    with pytest.raises(FileNotFoundError):
        takahe.load.from_file("data/file_that_doesnt_exist.dat")
