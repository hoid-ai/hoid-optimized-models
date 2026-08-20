import math
import os
import sys

# The pinned CUTLASS example tree is already on sys.path (the shim puts it
# there); the FMHA example itself lives in a subdirectory with no __init__.py,
# exactly as its own __main__ block assumes, so add it the way it does.
_root = next(p for p in sys.path if p.endswith("examples/python/CuTeDSL"))
_fmha_dir = os.path.join(_root, "cute/blackwell/kernel/attention/fmha")
if _fmha_dir not in sys.path:
    sys.path.insert(0, _fmha_dir)

from cutlass.cute.typing import Float32
import fmha as fmha_mod
from helpers import fmha_helpers as fmha_utils

# A residual mask is required when Skv does not fill the 128-wide N tile;
# a window mask is the cheaper path when it does.
_mask = (
    fmha_utils.MaskEnum.RESIDUAL_MASK
    if SKV % 128 != 0
    else fmha_utils.MaskEnum.WINDOW_MASK
)

_op = fmha_mod.BlackwellFusedMultiHeadAttentionForward(
    Float32,          # qk_acc_dtype
    Float32,          # pv_acc_dtype
    (128, 128),       # mma_tiler_mn — locked by the example
    D,
    True,             # is_persistent
    _mask,
    False,            # ex2
    True,             # skip_corr   (step 0's winner on every cohort)
    use_tma_store=True,
)

_scale = 1.0 / math.sqrt(D)
_problem = (B, SQ, SQ, SKV, H, H, D, D)


@cute.jit
def fmha_entry(
    o: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    stream: cuda.CUstream,
):
    _op(
        q, k, v, o,
        _problem,
        None, None, None, None,
        _scale * math.log2(math.e),
        _scale,
        1.0,
        None, None, None, None, None,
        stream,
        False,
    )
