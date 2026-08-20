import os
import sys

_root = next(p for p in sys.path if p.endswith("examples/python/CuTeDSL"))
_efc = os.path.join(_root, "cute/blackwell/efc")
if _efc not in sys.path:
    sys.path.insert(0, _efc)
from common_dense_gemm_efc import DenseGemmEFC

def _round_bf16(x):
    if isinstance(x, (int, float)):
        return x
    return x.to(cutlass.BFloat16).to(cutlass.Float32)

def _bias_store(efc_config, bias, D):
    # Match the incumbent: fp32 convolution accumulation plus fp32
    # bias, then exactly one bf16 rounding before exact-erf GELU.
    D.store(_round_bf16(efc_config.accum() + bias.load()))

_gemm = DenseGemmEFC(
    acc_dtype=cutlass.Float32,
    epi_dtype=cutlass.Float32,
    use_2cta_instrs=TWO_CTA,
    mma_tiler_mn=(TILE_M, TILE_N),
    cluster_shape_mn=(CLUSTER_M, CLUSTER_N),
    epilogue_function_configuration=_bias_store,
)

from cutlass.cute.runtime import make_fake_compact_tensor as _fake
_RM = (2, 1, 0)
_fake_bias = _fake(cutlass.BFloat16, (M, N, 1), stride_order=_RM)
_fake_d = _fake(cutlass.BFloat16, (M, N, 1), stride_order=_RM)
_gemm.efc.compile((_fake_bias, _fake_d))

@cute.jit
def conv2_blackwell_efc_bias_entry(
    d: cute.Tensor, x: cute.Tensor, w: cute.Tensor,
    bias: cute.Tensor, stream: cuda.CUstream,
):
    bias_layout = cute.make_layout((M, N, 1), stride=(0, 1, 0))
    bias_mn = cute.make_tensor(bias.iterator, bias_layout)
    _gemm(x, w, MAX_CLUSTERS, stream, (bias_mn, d))
