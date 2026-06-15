import json

import numcodecs
import numcodecs.registry
import numpy as np
import pytest


def test_filter_from_config():
    config = dict(id="replace.filter", replacements={})

    codec = numcodecs.registry.get_codec(config)
    assert codec.__class__.__name__ == "ReplaceFilterCodec"
    assert codec.__class__.__module__ == "numcodecs_replace"

    assert repr(codec) == "ReplaceFilterCodec(replacements={})"

    assert json.dumps(codec.get_config(), sort_keys=True) == json.dumps(
        config, sort_keys=True
    )


def test_meta_from_config():
    config = dict(
        id="replace.meta",
        replace=np.nan,
        with_=0,
        codec=dict(id="zlib", level=1),
        bitmap_codec=dict(id="packbits"),
    )

    codec = numcodecs.registry.get_codec(config)
    assert codec.__class__.__name__ == "ReplaceMetaCodec"
    assert codec.__class__.__module__ == "numcodecs_replace"

    assert (
        repr(codec)
        == "ReplaceMetaCodec(replace=nan, with_=0, codec=Zlib(level=1), bitmap_codec=PackBits())"
    )

    assert json.dumps(codec.get_config(), sort_keys=True) == json.dumps(
        config, sort_keys=True
    )


def check_filter_roundtrip(data: np.ndarray):
    config = dict(
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

    codec = numcodecs.registry.get_codec(config)

    assert (
        repr(codec)
        == "ReplaceFilterCodec(replacements={-inf: 'finite_min', 0: 'finite_mean', inf: 'finite_max', -1: 'nan_min', nan: 'nan_mean', 1: 'nan_max', 24: 42})"
    )

    assert json.dumps(codec.get_config(), sort_keys=True) == json.dumps(
        config, sort_keys=True
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
def test_filter_roundtrip():
    check_filter_roundtrip(np.zeros(tuple()))
    with pytest.warns(RuntimeWarning, match="empty slice"):
        check_filter_roundtrip(np.zeros((0,)))
    check_filter_roundtrip(np.arange(1000).reshape(10, 10, 10))
    check_filter_roundtrip(np.array([4.2, -2.4, np.nan, -np.nan, 0.0, -0.0]))
    check_filter_roundtrip(np.array([np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0]))
    check_filter_roundtrip(
        np.array(
            [np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0],
            dtype=np.dtype(np.float64).newbyteorder("<"),
        )
    )
    check_filter_roundtrip(
        np.array(
            [np.inf, -np.inf, np.nan, -np.nan, 0.0, -0.0],
            dtype=np.dtype(np.float64).newbyteorder(">"),
        )
    )


def _as_bits(a: np.ndarray) -> np.ndarray:
    return a.view(a.dtype.str.replace("f", "u").replace("i", "u"))
