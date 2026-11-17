"""Deprecated name of restoration.single_gap_closure (Phase 6).

Phase 6 computes no Elliptic Fourier Descriptors and was renamed to
symmetry-guided single-gap closure. This module re-exports everything from
the new one so that old imports keep working.
"""

import warnings

from restoration import single_gap_closure as _module
from restoration.single_gap_closure import *  # noqa: F401,F403

globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})

warnings.warn(
    "restoration.efd_closure is deprecated; import restoration.single_gap_closure",
    DeprecationWarning,
    stacklevel=2,
)
