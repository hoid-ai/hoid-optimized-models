import math
import os
import sys
from typing import Tuple
from cutlass.cute.typing import Float32

_root = next(p for p in sys.path if p.endswith("examples/python/CuTeDSL"))
_mi_dir = os.path.join(_root, "cute/blackwell/kernel/attention/mixed_input_fmha")
if _mi_dir not in sys.path:
    sys.path.insert(0, _mi_dir)
from cute.blackwell.kernel.attention.mixed_input_fmha import mixed_input_fmha_prefill_d512 as mi
from cute.blackwell.kernel.attention.mixed_input_fmha import prefill_helpers as prefill_utils
from helpers import fmha_helpers as fmha_utils

@cute.jit
def dequant_k_identity(
    iterations: int,
    transform_warp_ids: Tuple,
    dtype_args: Tuple,
    tensor_args: Tuple,
    pipeline_args: Tuple,
):
    (k_dtype, q_dtype) = dtype_args
    (sOrig, sScale, sTrans) = tensor_args
    (load_kv_consumer, load_scale_consumer, dequant_kv_producer) = pipeline_args
    tidx, _, _ = cute.arch.thread_idx()
    THREADS_PER_WARP = 32
    thread_idx = tidx % (THREADS_PER_WARP * len(transform_warp_ids))
    r2s_copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), k_dtype, num_bits_per_copy=32
    )
    # Construct tiled_copy satisfying 16 contiguous elts per copy atom
    r2s_tiled_copy = cute.make_cotiled_copy(
        r2s_copy_atom,
        cute.make_layout((256, 16), stride=(16, 1)),
        sTrans[(None, None, None, 0)].layout,
    )
    thr_r2s_tiled_copy = r2s_tiled_copy.get_slice(thread_idx)
    tOsOrig = thr_r2s_tiled_copy.partition_S(sOrig)
    tTsTrans = thr_r2s_tiled_copy.partition_D(sTrans)
    tOrOrig = cute.make_rmem_tensor_like(
        cute.append(
            tOsOrig[None, None, None, None, 0].layout,
            cute.make_layout(
                2, stride=cute.cosize(tOsOrig[None, None, None, None, 0].layout)
            ),
        ),
        k_dtype,
    )
    tTrTrans = cute.make_rmem_tensor_like(
        cute.append(
            tTsTrans[None, None, None, None, 0].layout,
            cute.make_layout(
                2, stride=cute.cosize(tTsTrans[None, None, None, None, 0].layout)
            ),
        ),
        q_dtype,
    )
    tSsScale = thr_r2s_tiled_copy.partition_S(sScale)
    tSrScale = cute.make_rmem_tensor_like(tSsScale[None, None, None, None, None, 0])
    scale_handle = load_scale_consumer.wait_and_advance()
    cute.autovec_copy(
        tSsScale[None, None, None, None, None, scale_handle.index], tSrScale
    )
    cute.arch.fence_view_async_shared()
    scale_handle.release()
    # prefetch iter = 0
    kv_handle = load_kv_consumer.wait_and_advance()
    cute.autovec_copy(
        tOsOrig[None, None, None, None, kv_handle.index],
        tOrOrig[None, None, None, None, 0],
    )
    transformed_tensor = tOrOrig[None, None, None, None, 0].load().to(q_dtype)
    scale = cute.TensorSSA(
        tSrScale[None, None, None, None, 0].load(),
        transformed_tensor.shape,
        q_dtype,
    )
    # identity BF16 path: scale intentionally ignored
    tTrTrans[None, None, None, None, 0].store(transformed_tensor)
    cute.arch.fence_view_async_shared()
    kv_handle.release()
    for iter in cutlass.range(1, iterations, unroll_full=True):
        kv_trans_handle = dequant_kv_producer.acquire_and_advance()
        cute.autovec_copy(
            tTrTrans[None, None, None, None, (iter - 1) % 2],
            tTsTrans[None, None, None, None, kv_trans_handle.index],
        )
        cute.arch.fence_view_async_shared()
        kv_trans_handle.commit()
        kv_handle = load_kv_consumer.wait_and_advance()
        cute.autovec_copy(
            tOsOrig[None, None, None, None, kv_handle.index],
            tOrOrig[None, None, None, None, iter % 2],
        )
        transformed_tensor = (
            tOrOrig[None, None, None, None, iter % 2].load().to(q_dtype)
        )
        scale = cute.TensorSSA(
            tSrScale[None, None, None, None, iter].load(),
            transformed_tensor.shape,
            q_dtype,
        )
        # identity BF16 path: scale intentionally ignored
        tTrTrans[None, None, None, None, iter % 2].store(transformed_tensor)
        cute.arch.fence_view_async_shared()
        kv_handle.release()
    kv_trans_handle = dequant_kv_producer.acquire_and_advance()
    cute.autovec_copy(
        tTrTrans[None, None, None, None, (iterations - 1) % 2],
        tTsTrans[None, None, None, None, kv_trans_handle.index],
    )
    cute.arch.fence_view_async_shared()
    kv_trans_handle.commit()
    return load_kv_consumer, load_scale_consumer, dequant_kv_producer



@cute.jit
def dequant_v_identity(
    iterations: int,
    transform_warp_ids: Tuple,
    dtype_args: Tuple,
    tensor_args: Tuple,
    pipeline_args: Tuple,
):
    (v_dtype, q_dtype) = dtype_args
    (sOrig, sScale, sTrans) = tensor_args
    (load_kv_consumer, load_scale_consumer, dequant_kv_producer) = pipeline_args
    tidx, _, _ = cute.arch.thread_idx()
    THREADS_PER_WARP = 32
    thread_idx = tidx % (THREADS_PER_WARP * len(transform_warp_ids))
    r2s_copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), v_dtype, num_bits_per_copy=32
    )
    # Construct tiled_copy satisfying 16 contiguous elts per copy atom
    r2s_tiled_copy = cute.make_cotiled_copy(
        r2s_copy_atom,
        cute.make_layout((256, 16), stride=(16, 1)),
        sTrans[(None, None, None, 0)].layout,
    )
    thr_r2s_tiled_copy = r2s_tiled_copy.get_slice(thread_idx)
    tOsOrig = thr_r2s_tiled_copy.partition_S(sOrig)
    tTsTrans = thr_r2s_tiled_copy.partition_D(sTrans)
    # double buffer for better perf
    tOrOrig = cute.make_rmem_tensor_like(
        cute.append(
            tOsOrig[None, None, None, None, 0].layout,
            cute.make_layout(
                2, stride=cute.cosize(tOsOrig[None, None, None, None, 0].layout)
            ),
        ),
        v_dtype,
    )
    tTrTrans = cute.make_rmem_tensor_like(
        cute.append(
            tTsTrans[None, None, None, None, 0].layout,
            cute.make_layout(
                2, stride=cute.cosize(tTsTrans[None, None, None, None, 0].layout)
            ),
        ),
        q_dtype,
    )
    tSsScale = thr_r2s_tiled_copy.partition_S(sScale)
    tSrScale = cute.make_rmem_tensor_like(tSsScale[None, None, None, None, None, 0])
    scale_v_handle = load_scale_consumer.wait_and_advance()
    cute.autovec_copy(
        tSsScale[None, None, None, None, None, scale_v_handle.index],
        tSrScale,
    )
    cute.arch.fence_view_async_shared()
    scale_v_handle.release()
    # prefetch iter = 0
    kv_handle = load_kv_consumer.wait_and_advance()
    cute.autovec_copy(
        tOsOrig[None, None, None, None, kv_handle.index],
        tOrOrig[None, None, None, None, 0],
    )
    transformed_tensor = tOrOrig[None, None, None, None, 0].load().to(q_dtype)
    scale = cute.TensorSSA(
        tSrScale[None, None, None, None, 0].load(),
        transformed_tensor.shape,
        q_dtype,
    )
    # identity BF16 path: scale intentionally ignored
    tTrTrans[None, None, None, None, 0].store(transformed_tensor)
    cute.arch.fence_view_async_shared()
    kv_handle.release()
    for iter in cutlass.range(1, iterations, unroll_full=True):
        kv_trans_handle = dequant_kv_producer.acquire_and_advance()
        cute.autovec_copy(
            tTrTrans[None, None, None, None, (iter - 1) % 2],
            tTsTrans[None, None, None, None, kv_trans_handle.index],
        )
        cute.arch.fence_view_async_shared()
        kv_trans_handle.commit()
        kv_handle = load_kv_consumer.wait_and_advance()
        cute.autovec_copy(
            tOsOrig[None, None, None, None, kv_handle.index],
            tOrOrig[None, None, None, None, iter % 2],
        )
        transformed_tensor = (
            tOrOrig[None, None, None, None, iter % 2].load().to(q_dtype)
        )
        scale = cute.TensorSSA(
            tSrScale[
                None,
                None,
                None,
                None,
                iter,
            ].load(),
            transformed_tensor.shape,
            q_dtype,
        )
        # identity BF16 path: scale intentionally ignored
        tTrTrans[None, None, None, None, iter % 2].store(transformed_tensor)
        cute.arch.fence_view_async_shared()
        kv_handle.release()
    kv_trans_handle = dequant_kv_producer.acquire_and_advance()
    cute.autovec_copy(
        tTrTrans[None, None, None, None, (iterations - 1) % 2],
        tTsTrans[None, None, None, None, kv_trans_handle.index],
    )
    cute.arch.fence_view_async_shared()
    kv_trans_handle.commit()
    return load_kv_consumer, load_scale_consumer, dequant_kv_producer

# Replace only this authored module's helper bindings: preserve pipeline
# acquire/release structure but make BF16 K/V transformation identity.
prefill_utils.dequant_k = dequant_k_identity
prefill_utils.dequant_v = dequant_v_identity

class BF16D512(mi.MixedInputFusedMultiHeadAttentionPrefillD512):
    def _setup_attributes(self):
        super()._setup_attributes()
        self.kv_stage = 2

_op = BF16D512(
    512, Float32, Float32, False, fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
)
_problem = (1, 16384, 16384, 1, 1, 512)
_scale = 1.0 / math.sqrt(512.0)

@cute.jit
def fmha_d512_bf16_np_entry(o: cute.Tensor, q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, stream: cuda.CUstream):
    _op(
        q.iterator, k.iterator, v.iterator, o.iterator,
        k.iterator, v.iterator,
        _problem, _scale * math.log2(math.e), 1.0,
        None, None, stream,
    )
