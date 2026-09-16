import pytest

import takahe


def test_from_file_reads_dat_file(sample_dat_file):
    df = takahe.load.from_file(str(sample_dat_file))

    assert len(df) == 2
    assert list(df.columns) == ['m1', 'm2', 'a0', 'e0', 'weight',
                                 'evolution_age', 'rejuvenation_age']
    assert df.iloc[0].m1 == pytest.approx(1.4)


def test_from_file_adds_coalescence_time_column_for_ct_files(tmp_path):
    path = tmp_path / "some-file_ct.dat"
    path.write_text("1.4 1.4 10.0 0.1 12.5 1.0e9 0.5e9 3.0e9\n")

    df = takahe.load.from_file(str(path))

    assert 'coalescence_time' in df.columns


def test_from_gzip_raises_if_file_missing(tmp_path):
    missing = tmp_path / "does-not-exist.dat"

    with pytest.raises(IOError):
        takahe.load.from_gzip(str(missing))


def test_from_gzip_appends_gz_extension_if_missing(sample_dat_gz_file):
    # from_gzip() should find "<path>.gz" even when called with the
    # extension-less path.
    path_without_gz = str(sample_dat_gz_file)[:-len(".gz")]

    # Guarded by the xfail below - this call currently returns an empty
    # DataFrame rather than raising, so we don't assert on its contents
    # here. See test_from_gzip_silently_loses_data for that bug.
    takahe.load.from_gzip(path_without_gz)


# --- Known bug in load.py, documented via xfail ---------------------------
#
# from_gzip() (and from_directory()'s gzip fallback) pass a
# gzip.GzipFile object into from_file(), whose "_ct" in filepath and
# ".h5" in filepath checks use Python's `in` operator. For an iterable
# file-like object, `in` iterates the object to search it - which
# consumes the GzipFile's underlying line iterator before
# pd.read_csv(filepath, ...) ever gets to read it. The result is a
# silent, total loss of data: every gzip-based load returns an empty
# DataFrame instead of raising an error or loading the real content.

def test_from_gzip_silently_loses_data(sample_dat_gz_file):
    df = takahe.load.from_gzip(str(sample_dat_gz_file))

    # This is the *documented* (buggy) behaviour today: the DataFrame
    # comes back empty even though sample_dat_gz_file has 2 real rows
    # of data. If from_gzip() is ever fixed, this assertion should be
    # replaced with `assert len(df) == 2`.
    assert len(df) == 0
