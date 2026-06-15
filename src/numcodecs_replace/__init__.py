"""
Value replacement codecs for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["ReplaceFilterCodec", "ReplaceMetaCodec", "Replacement"]

from collections.abc import Callable
from enum import Enum, auto
from functools import reduce
from io import BytesIO
from typing import Literal, TypeVar

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
from numcodecs.abc import Codec
from numcodecs_combinators.abc import CodecCombinatorMixin
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

    Parameters
    ----------
    replacements : dict[int | float, int | float | Replacement | Literal["finite_min", "finite_mean", "finite_max", "nan_min", "nan_mean", "nan_max"]]
        Mapping from values to be replaced to the replacement values.
    """

    __slots__: tuple[str, ...] = ("_replacements",)
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
                k: (v.name if isinstance(v, Replacement) else v)
                for k, v in self._replacements.items()
            },
        )

    def __repr__(self) -> str:
        replacements = {
            k: (v.name if isinstance(v, Replacement) else v)
            for k, v in self._replacements.items()
        }
        return f"{type(self).__name__}(replacements={replacements!r})"


numcodecs.registry.register_codec(ReplaceFilterCodec)


class ReplaceMetaCodec(Codec, CodecCombinatorMixin):
    """
    Meta-codec that replaces a value during encoding and restores it during decoding.

    The values-after-replacement are encoded with the `codec`, the
    [boolean][numpy.bool] mask of where values were replaced is encoded with
    the `bitmap_codec`.

    The special [`Replacement`][..Replacement] values, e.g.
    [`Replacement.nan_mean`][..Replacement.nan_mean], are derived from the data.
    Multiple [`ReplaceMetaCodec`][.]s can be stacked, e.g. using the
    [`numcodecs-combinators`](https://numcodecs-combinators.readthedocs.io)
    package, to apply some replacements before computing e.g. the finite mean.

    When replacing NaN values, all values that are NaN are replaced,
    irrespective of their bitpatterns.

    Parameters
    ----------
    replace : int | float
        The value to be replaced.
    with_ : int | float | Replacement | Literal["finite_min", "finite_mean", "finite_max", "nan_min", "nan_mean", "nan_max"]
        The replacement value.
    codec : dict | Codec
        The configuration or instantiated codec that encodes the values after
        replacement.
    bitmap_codec : dict | Codec
        The configuration or instantiated codec that encodes the
        [boolean][numpy.bool] mask of where values were replaced.

        For instance, the [`numcodecs.PackBits`][numcodecs.packbits.PackBits]
        codec can be used to pack the mask into a byte array.
    """

    __slots__: tuple[str, ...] = ("_replace", "_with", "_codec", "_bitmap_codec")
    _replace: int | float
    _with: int | float | Replacement
    _codec: Codec
    _bitmap_codec: Codec

    codec_id: str = "replace.meta"  # type: ignore

    def __init__(
        self,
        *,
        replace: int | float,
        with_: int
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
        codec: dict | Codec,
        bitmap_codec: dict | Codec,
    ) -> None:
        self._replace = replace
        self._with = Replacement[with_] if isinstance(with_, str) else with_
        self._codec = (
            codec if isinstance(codec, Codec) else numcodecs.registry.get_codec(codec)
        )
        self._bitmap_codec = (
            bitmap_codec
            if isinstance(bitmap_codec, Codec)
            else numcodecs.registry.get_codec(bitmap_codec)
        )

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
        dtype, shape = a.dtype, a.shape

        with_ = (
            self._with.compute(a) if isinstance(self._with, Replacement) else self._with
        )

        if isinstance(self._replace, int) or not np.isnan(self._replace):
            is_replace = a == self._replace
        else:
            is_replace = np.isnan(a)
        a[is_replace] = with_

        # message: dtype shape encoded-dtype encoded-shape [padding] encoded
        #          bitmap-dtype bitmap-shape [padding] bitmap
        message: list[bytes | bytearray] = []

        message.append(leb128.u.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(shape)))
        for s in shape:
            message.append(leb128.u.encode(s))

        encoded = self._codec.encode(a)
        encoded = numcodecs.compat.ensure_ndarray(encoded)

        message.append(leb128.u.encode(len(encoded.dtype.str)))
        message.append(encoded.dtype.str.encode("ascii"))

        message.append(leb128.u.encode(encoded.ndim))
        for s in encoded.shape:
            message.append(leb128.u.encode(s))

        # insert padding to align with encoded itemsize
        message.append(
            b"\0"
            * (
                encoded.dtype.itemsize
                - (sum(len(m) for m in message) % encoded.itemsize)
            )
        )

        # ensure that the encoded values are encoded in little endian binary
        message.append(encoded.astype(encoded.dtype.newbyteorder("<")).tobytes())

        bitmap = self._bitmap_codec.encode(is_replace)
        bitmap = numcodecs.compat.ensure_ndarray(bitmap)

        message.append(leb128.u.encode(len(bitmap.dtype.str)))
        message.append(bitmap.dtype.str.encode("ascii"))

        message.append(leb128.u.encode(bitmap.ndim))
        for s in bitmap.shape:
            message.append(leb128.u.encode(s))

        # insert padding to align with bitmap itemsize
        message.append(
            b"\0"
            * (bitmap.dtype.itemsize - (sum(len(m) for m in message) % bitmap.itemsize))
        )

        # ensure that the bitmap values are encoded in little endian binary
        message.append(bitmap.astype(bitmap.dtype.newbyteorder("<")).tobytes())

        return b"".join(message)

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

        b = numcodecs.compat.ensure_bytes(buf)
        b_io = BytesIO(b)

        # message: dtype shape encoded-dtype encoded-shape [padding] encoded
        #          bitmap-dtype bitmap-shape [padding] bitmap
        dtype = np.dtype(b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii"))
        shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )

        encoded_dtype = np.dtype(
            b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
        )
        encoded_shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        encoded_size = reduce(lambda a, b: a * b, encoded_shape, 1)

        # remove padding to align with encoded itemsize
        b_io.read(encoded_dtype.itemsize - (b_io.tell() % encoded_dtype.itemsize))

        encoded = (
            np.frombuffer(
                b_io.read(encoded_size * encoded_dtype.itemsize),
                dtype=encoded_dtype.newbyteorder("<"),
                count=encoded_size,
            )
            .astype(encoded_dtype)
            .reshape(encoded_shape)
        )

        decoded = np.empty(shape, dtype=dtype)
        self._codec.decode(encoded, out=decoded)

        bitmap_dtype = np.dtype(
            b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
        )
        bitmap_shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        bitmap_size = reduce(lambda a, b: a * b, bitmap_shape, 1)

        # remove padding to align with bitmap itemsize
        b_io.read(bitmap_dtype.itemsize - (b_io.tell() % bitmap_dtype.itemsize))

        bitmap = (
            np.frombuffer(
                b_io.read(bitmap_size * bitmap_dtype.itemsize),
                dtype=bitmap_dtype.newbyteorder("<"),
                count=bitmap_size,
            )
            .astype(bitmap_dtype)
            .reshape(bitmap_shape)
        )

        is_replace = np.empty(shape, dtype=np.bool)
        self._bitmap_codec.decode(bitmap, out=is_replace)

        decoded[is_replace] = self._replace

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

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
            replace=self._replace,
            with_=self._with.name
            if isinstance(self._with, Replacement)
            else self._with,
            codec=self._codec.get_config(),
            bitmap_codec=self._bitmap_codec.get_config(),
        )

    def __repr__(self) -> str:
        with_ = self._with.name if isinstance(self._with, Replacement) else self._with
        return f"{type(self).__name__}(replace={self._replace!r}, with_={with_!r}, codec={self._codec!r}, bitmap_codec={self._bitmap_codec!r})"

    def map(self, mapper: Callable[[Codec], Codec]) -> "ReplaceMetaCodec":
        """
        Apply the `mapper` to this replacement meta-codec.

        In the returned [`ReplaceMetaCodec`][..], the `codec` and
        `bitmap_codec` are replaced by their mapped codecs.

        The `mapper` should recursively apply itself to any inner codecs that
        also implement the
        [`CodecCombinatorMixin`][numcodecs_combinators.abc.CodecCombinatorMixin]
        mixin.

        To automatically handle the recursive application as a caller, you can
        use
        ```python
        numcodecs_combinators.map_codec(codec, mapper)
        ```
        instead.

        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that should be applied to the wrapped `codec` and
            `bitmap_codec` to map over this replacement meta-codec.

        Returns
        -------
        mapped : ReplaceMetaCodec
            The mapped replacement meta-codec.
        """

        return ReplaceMetaCodec(
            replace=self._replace,
            with_=self._with,
            codec=mapper(self._codec),
            bitmap_codec=mapper(self._bitmap_codec),
        )


numcodecs.registry.register_codec(ReplaceMetaCodec)
