import warnings

from lineage_evo.qlib_warnings import suppress_qlib_all_nan_slice_warning


def test_suppresses_only_all_nan_slice_runtime_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with suppress_qlib_all_nan_slice_warning():
            warnings.warn("All-NaN slice encountered", RuntimeWarning)
            warnings.warn("other runtime warning", RuntimeWarning)

    messages = [str(item.message) for item in caught]
    assert "All-NaN slice encountered" not in messages
    assert "other runtime warning" in messages
