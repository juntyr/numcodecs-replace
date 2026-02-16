import numcodecs
import numcodecs.registry
import numpy as np
import pytest


def test_from_config():
    codec = numcodecs.registry.get_codec(dict(id="replace.filter", replacements={}))
    assert codec.__class__.__name__ == "ReplaceFilterCodec"
    assert codec.__class__.__module__ == "numcodecs_replace"


def check_roundtrip(data: np.ndarray):
    codec = numcodecs.registry.get_codec(
        dict(
            id="replace.filter",
            replacements={
                -np.inf: "finite_min",
                0: "finite_mean",
                +np.inf: "finite_max",
                -1: "nan_min",
                np.nan: "nan_mean",
                +1: "nan_max",
                24: 42,
            },
        )
    )

    encoded = codec.encode(data)

    assert encoded.dtype == data.dtype
    assert encoded.shape == data.shape

    is_floating = np.issubdtype(data.dtype, np.floating)
    info = np.finfo(data.dtype) if is_floating else np.iinfo(data.dtype)  # type: ignore

    assert np.all(
        (data != -np.inf)
        | (encoded == np.amin(data, where=np.isfinite(data), initial=info.max))
    )
    assert np.all(
        (data != 0)
        | (
            encoded
            == np.array(np.mean(data, where=np.isfinite(data))).astype(data.dtype)
        )
    )
    assert np.all(
        (data != np.inf)
        | (encoded == np.amax(data, where=np.isfinite(data), initial=info.min))
    )

    assert np.all(
        (data != -1)
        | (encoded == np.nanmin(data, initial=(np.inf if is_floating else info.max)))
    )
    if np.isnan(np.nanmean(data)):
        assert np.all(~np.isnan(data) | np.isnan(encoded))
    else:
        assert np.all(
            ~np.isnan(data) | (encoded == np.array(np.nanmean(data)).astype(data.dtype))
        )
    assert np.all(
        (data != +1)
        | (encoded == np.nanmax(data, initial=(-np.inf if is_floating else info.min)))
    )

    assert np.all((data != 24) | (encoded == 42))

    decoded = codec.decode(encoded)

    assert decoded.dtype == data.dtype
    assert decoded.shape == data.shape

    assert np.all(_as_bits(decoded) == _as_bits(encoded))


@np.errstate(invalid="ignore")
def test_roundtrip():
    check_roundtrip(np.zeros(tuple()))
    with pytest.warns(RuntimeWarning, match="empty slice"):
        check_roundtrip(np.zeros((0,)))
    check_roundtrip(np.arange(1000).reshape(10, 10, 10))
    check_roundtrip(np.array([np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0]))
    check_roundtrip(
        np.array(
            [np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0],
            dtype=np.dtype(np.float64).newbyteorder("<"),
        )
    )
    check_roundtrip(
        np.array(
            [np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0],
            dtype=np.dtype(np.float64).newbyteorder(">"),
        )
    )


def _as_bits(a: np.ndarray) -> np.ndarray:
    return a.view(a.dtype.str.replace("f", "u").replace("i", "u"))
