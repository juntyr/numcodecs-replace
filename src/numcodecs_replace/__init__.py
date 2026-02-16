"""
Value replacement codecs for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["ReplaceFilterCodec", "Replacement"]

from enum import Enum, auto
from typing import Literal, TypeVar

import numcodecs.compat
import numcodecs.registry
import numpy as np
from numcodecs.abc import Codec
from typing_extensions import (
    Buffer,  # MSPV 3.12
    assert_never,  # MSPV 3.11
)

S = TypeVar("S", bound=tuple[int, ...])
""" Any array shape. """

T = TypeVar("T", bound=np.number)
""" Any numeric type. """


class Replacement(Enum):
    """
    Special replacement values that are derived from the original data.
    """

    finite_min = auto()
    """ The finite minimum of the data, or the largest-possible finite value if the data contains no finite values. """

    finite_mean = auto()
    """ The finite mean of the data, or NaN if the data contains no finite values. """

    finite_max = auto()
    """ The finite maximum of the data, or the smallest-possible finite value if the data contains no finite values. """

    nan_min = auto()
    """ The non-NaN minimum of the data, or the largest-possible value if the data contains no non-NaN values. """

    nan_mean = auto()
    """ The non-NaN mean of the data, or NaN if the data contains no non-NaN values. """

    nan_max = auto()
    """ The non-NaN maximum of the data, or the smallest-possible value if the data contains no non-NaN values. """

    def compute(self, x: np.ndarray[S, np.dtype[T]]) -> T:
        """
        Compute the special replacement value for the array `x`.

        Parameters
        ----------
        x : np.ndarray[S, np.dtype[T]]
            The numerical input data array.

        Returns
        -------
        replacement : T
            The scalar replacement value.
        """

        is_floating = np.issubdtype(x.dtype, np.floating)
        info = np.finfo(x.dtype) if is_floating else np.iinfo(x.dtype)  # type: ignore

        match self:
            case Replacement.finite_min:
                return np.amin(x, initial=info.max, where=np.isfinite(x))
            case Replacement.finite_mean:
                return np.array(np.mean(x, where=np.isfinite(x))).astype(x.dtype)[()]  # type: ignore
            case Replacement.finite_max:
                return np.amax(x, initial=info.min, where=np.isfinite(x))
            case Replacement.nan_min:
                return np.nanmin(x, initial=(np.inf if is_floating else info.max))
            case Replacement.nan_mean:
                return np.array(np.nanmean(x)).astype(x.dtype)[()]  # type: ignore
            case Replacement.nan_max:
                return np.nanmax(x, initial=(-np.inf if is_floating else info.min))
            case _:
                assert_never(self)


class ReplaceFilterCodec(Codec):
    """
    Filter codec that replaces configured values during encoding and passes through the data during decoding.

    The replacements are processed in order.

    The special [`Replacement`][..Replacement] values, e.g.
    [`Replacement.nan_mean`][..Replacement.nan_mean], are derived from the data
    before any replacements are made.
    Multiple [`ReplaceFilterCodec`][.]s can be stacked, e.g. using the
    [`numcodecs-combinators`](https://numcodecs-combinators.readthedocs.io)
    package, to apply some replacements before computing e.g. the finite mean.

    When replacing NaN values, all values that are NaN are replaced,
    irrespective of their bitpatterns.
    """

    __slots__ = "_replacements"
    _replacements: dict[
        int | float,
        int | float | Replacement,
    ]

    codec_id: str = "replace.filter"  # type: ignore

    def __init__(
        self,
        *,
        replacements: dict[
            int | float,
            int
            | float
            | Replacement
            | Literal[
                "finite_min",
                "finite_mean",
                "finite_max",
                "nan_min",
                "nan_mean",
                "nan_max",
            ],
        ],
    ) -> None:
        self._replacements = {
            k: (Replacement[v] if isinstance(v, str) else v)
            for k, v in replacements.items()
        }

    def encode(self, buf: Buffer) -> Buffer:
        """Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        """

        a = np.copy(numcodecs.compat.ensure_ndarray(buf))

        replacements: dict[int | float, int | float | np.number] = {
            k: (v.compute(a) if isinstance(v, Replacement) else v)
            for k, v in self._replacements.items()
        }

        for k, v in replacements.items():
            if isinstance(k, int) or not np.isnan(k):
                a[a == k] = v
            else:
                a[np.isnan(a)] = v

        return a  # type: ignore

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol.
        """

        return numcodecs.compat.ndarray_copy(buf, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of this replacement filter codec.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this codec from the returned config.

        Returns
        -------
        config : dict
            Configuration of this replacement filter codec.
        """

        return dict(
            id=type(self).codec_id,
            replacements={
                k: (str(v) if isinstance(v, Replacement) else v)
                for k, v in self._replacements.items()
            },
        )


numcodecs.registry.register_codec(ReplaceFilterCodec)
