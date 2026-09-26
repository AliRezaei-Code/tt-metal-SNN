# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Canonical constructors for TT-Metalium program descriptors.

Every SNN program is the same three kinds of object -- a kernel descriptor, a circular-buffer
descriptor, and a runtime-argument block -- so they are built here once. Duplicated per module
they drift, and they already had: the accessor-offset fix below was applied to the reader and
writer argument lists independently in two files before being pulled out.

``ttnn`` is imported inside each function, not at module scope, so importing this file (and
therefore ``snn.layout``'s consumers, and the device-free reference tests) does not require an
installed tt-metal. A checkout that happens to have a ``ttnn/`` directory on ``sys.path`` will
import it as an empty namespace package and then fail on first use, which is worse than failing
at import.
"""

from models.experimental.snn.snn.layout import CB_DTYPE, CB_PAGE_BYTES, CB_TOTAL_BYTES


def cb_descriptor(index, core_ranges):
    """A circular-buffer descriptor for the buffer at ``index`` in the shared CB map.

    The data format and the page size both come from ``layout``, which is what this module has
    always claimed: the map is the single owner of the buffer geometry, and a caller that supplied
    either would be free to disagree with the kernel that reads the buffer. The signature used to
    take both as arguments, so ``total_size`` came from the table while ``page_size`` and
    ``data_format`` came from the call site -- the documented contract and the code disagreed, and a
    buffer could be declared two tiles deep carrying the page size of a different dtype. Nothing
    compared the two; it would have surfaced only as a mis-shaped buffer at dispatch.

    All eighteen call sites passed exactly what the table supplies, so deriving both changes no
    descriptor. It removes the possibility instead, and needs no guard, because there is nothing
    left to guard.
    """
    import ttnn

    data_format = {"float32": ttnn.float32, "bfloat16": ttnn.bfloat16, "uint32": ttnn.uint32}[CB_DTYPE[index]]
    return ttnn.CBDescriptor(
        total_size=CB_TOTAL_BYTES[index],
        core_ranges=core_ranges,
        format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=CB_PAGE_BYTES[index])
        ],
    )


def runtime_args(grid, args_for):
    """One runtime-argument entry per core in ``grid``, values supplied by ``args_for``.

    A kernel created on a ``CoreRangeSet`` runs on *every* core in it, and a core with no entry
    is undefined behaviour rather than a no-op -- so the loop is over the whole grid, not over the
    cores that happen to need different arguments. The single-core case is this function over a
    1x1 grid with a constant callback; there is deliberately no second, "simpler" helper,
    because two variants of the same rule is how they drift.

    ``ttnn.RuntimeArgs`` has no ``__add__``: the two bound forms are ``rtargs[x][y] = [...]`` and
    ``rtargs.append(coord, [...])``. This uses the second, which is what handles a grid.
    """
    import ttnn

    args = ttnn.RuntimeArgs()
    for core in ttnn.corerange_to_cores(grid):
        args.append(core, args_for(core))
    return args


def dm_compile_time_args(cb_args, *tensors):
    """Compile-time args for a data-movement kernel: offset, CB indices, then the accessors.

    The layout is ``[accessor_offset, *cb_args, *TensorAccessor blocks]``. ``TensorAccessorArgs<N>``
    reads its fields from compile-time argument ``N`` onward, so the offset has to skip the CB
    indices *and itself* -- hence ``1 + len(cb_args)``. Passing ``len(cb_args)`` would make the
    first accessor read the offset as its config word, shifting every field after it.
    """
    import ttnn

    return (
        [1 + len(cb_args)]
        + list(cb_args)
        + [arg for tensor in tensors for arg in ttnn.TensorAccessorArgs(tensor).get_compile_time_args()]
    )
