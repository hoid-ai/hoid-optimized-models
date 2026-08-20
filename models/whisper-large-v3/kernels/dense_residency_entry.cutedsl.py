import os
import sys

_root = next(p for p in sys.path if p.endswith("examples/python/CuTeDSL"))
_dense = os.path.join(_root, "cute/blackwell/kernel/dense_gemm")
if _dense not in sys.path:
    sys.path.insert(0, _dense)
from dense_gemm_persistent import PersistentDenseGemmKernel

_gemm = PersistentDenseGemmKernel(
    acc_dtype=cutlass.Float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(TILE_M, TILE_N),
    cluster_shape_mn=(1, 1),
    use_tma_store=True,
)
# The upstream default is occupancy=1, which deliberately consumes nearly
# the whole SM (230 KB here). Raising this target makes _compute_stages
# budget A/B and C staging per resident CTA instead of per SM.
_gemm.occupancy = OCCUPANCY

@cute.jit
def dense_residency_entry_tilem128tilen128occupancy1maxclusters148(
    out: cute.Tensor, x: cute.Tensor, w: cute.Tensor,
    stream: cuda.CUstream,
):
    _gemm(x, w, out, MAX_CLUSTERS, stream)

# CuTeDSL sweep materialization gives the decorated function a unique
# per-config suffix. Keep the static ABI name requested by cutedsl.entry
# as an alias to that renamed function. Split string literals prevent the
# whole-identifier renamer from rewriting the alias key as well.
globals()["dense_" + "residency_entry"] = dense_residency_entry_tilem128tilen128occupancy1maxclusters148
