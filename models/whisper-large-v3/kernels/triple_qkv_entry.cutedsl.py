# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
"""Blackwell persistent triple-GEMM QKV projection for Hoid.

Derived from the measured in-tree dual_geglu.py kernel: one A TMA stream,
three B streams, three disjoint TMEM accumulators, and three TMA-store
epilogues. Q/V preserve the incumbent bf16-round, bias-add, bf16-round order.
"""
import os
import sys
from typing import Optional, Tuple, Type, Union

import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.arch.constants import WARP_SIZE
from cutlass.cutlass_dsl import Boolean
import cutlass.utils as utils

_root = next(p for p in sys.path if p.endswith("examples/python/CuTeDSL"))
_dense = os.path.join(_root, "cute/blackwell/kernel/dense_gemm")
if _dense not in sys.path:
    sys.path.insert(0, _dense)
from dense_gemm_persistent import PersistentDenseGemmKernel
from cutlass.utils.gemm.sm100 import (
    transform_partitioned_tensor_layout,
    epilogue_tmem_copy_and_partition,
    epilogue_smem_copy_and_partition,
)


@cute.jit
def triple_qkv_epilogue_tma(
    gemm_kernel, epi_tidx, warp_idx,
    tma_q, tma_k, tma_v,
    acc_q_base, acc_k_base, acc_v_base,
    sC, gQ_base, gK_base, gV_base, bias_q_base, bias_v_base,
    epi_tile, num_tiles_executed, coord, state, acc_pipeline, c_pipeline,
):
    """Store three TMEM accumulators, preserving incumbent bf16 bias rounding."""
    gQ = transform_partitioned_tensor_layout(gQ_base)
    gK = transform_partitioned_tensor_layout(gK_base)
    gV = transform_partitioned_tensor_layout(gV_base)
    bQ = transform_partitioned_tensor_layout(bias_q_base)
    bV = transform_partitioned_tensor_layout(bias_v_base)
    aQ = transform_partitioned_tensor_layout(acc_q_base)
    aK = transform_partitioned_tensor_layout(acc_k_base)
    aV = transform_partitioned_tensor_layout(acc_v_base)
    copy_t2r, rQ_base, rQ = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, aQ, gQ, epi_tile, gemm_kernel.use_2cta_instrs)
    _, rK_base, rK = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, aK, gK, epi_tile, gemm_kernel.use_2cta_instrs)
    _, rV_base, rV = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, aV, gV, epi_tile, gemm_kernel.use_2cta_instrs)
    rOut = cute.make_rmem_tensor(rQ.shape, gemm_kernel.c_dtype)
    copy_r2s, rOut_view, sC_view = epilogue_smem_copy_and_partition(
        gemm_kernel, copy_t2r, rOut, epi_tidx, sC)

    q_epi = cute.flat_divide(gQ, epi_tile)
    q_s, q_g_part = cpasync.tma_partition(
        tma_q, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2),
        cute.group_modes(q_epi, 0, 2))
    q_g = q_g_part[(None, None, None, *coord)]
    k_epi = cute.flat_divide(gK, epi_tile)
    k_s, k_g_part = cpasync.tma_partition(
        tma_k, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2),
        cute.group_modes(k_epi, 0, 2))
    k_g = k_g_part[(None, None, None, *coord)]
    v_epi = cute.flat_divide(gV, epi_tile)
    v_s, v_g_part = cpasync.tma_partition(
        tma_v, 0, cute.make_layout(1), cute.group_modes(sC, 0, 2),
        cute.group_modes(v_epi, 0, 2))
    v_g = v_g_part[(None, None, None, *coord)]

    owner = copy_t2r.get_slice(epi_tidx)
    rbQ_part = owner.partition_D(cute.flat_divide(bQ, epi_tile))
    rbV_part = owner.partition_D(cute.flat_divide(bV, epi_tile))
    rbQ = cute.make_rmem_tensor(rQ.shape, gemm_kernel.c_dtype)
    rbV = cute.make_rmem_tensor(rQ.shape, gemm_kernel.c_dtype)
    gbQ = rbQ_part[(None, None, None, None, None, *coord)]
    gbV = rbV_part[(None, None, None, None, None, *coord)]

    barrier = pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id,
        num_threads=WARP_SIZE * len(gemm_kernel.epilogue_warp_id))
    rQ_src = rQ_base[(None, None, None, None, None, state.index)]
    rK_src = rK_base[(None, None, None, None, None, state.index)]
    rV_src = rV_base[(None, None, None, None, None, state.index)]
    acc_pipeline.consumer_wait(state)
    rQ_src = cute.group_modes(rQ_src, 3, cute.rank(rQ_src))
    rK_src = cute.group_modes(rK_src, 3, cute.rank(rK_src))
    rV_src = cute.group_modes(rV_src, 3, cute.rank(rV_src))
    gbQ = cute.group_modes(gbQ, 3, cute.rank(gbQ))
    gbV = cute.group_modes(gbV, 3, cute.rank(gbV))
    q_g = cute.group_modes(q_g, 1, cute.rank(q_g))
    k_g = cute.group_modes(k_g, 1, cute.rank(k_g))
    v_g = cute.group_modes(v_g, 1, cute.rank(v_g))
    subtiles = cute.size(rQ_src.shape, mode=[3])
    base_seq = num_tiles_executed * subtiles * 3
    for sub in range(subtiles):
        cute.copy(copy_t2r, rQ_src[(None, None, None, sub)], rQ)
        cute.copy(copy_t2r, rK_src[(None, None, None, sub)], rK)
        cute.copy(copy_t2r, rV_src[(None, None, None, sub)], rV)
        cute.autovec_copy(gbQ[(None, None, None, sub)], rbQ)
        cute.autovec_copy(gbV[(None, None, None, sub)], rbV)
        # Q: fp32 accumulator -> bf16, bf16 bias add -> bf16.
        x = copy_r2s.retile(rQ).load().to(gemm_kernel.c_dtype)
        bias = copy_r2s.retile(rbQ).load()
        rOut_view.store((x + bias).to(gemm_kernel.c_dtype))
        buf = (base_seq + sub * 3) % gemm_kernel.num_c_stage
        cute.copy(copy_r2s, rOut_view, sC_view[(None, None, None, buf)])
        cute.arch.fence_proxy("async.shared", space="cta")
        barrier.arrive_and_wait()
        if warp_idx == gemm_kernel.epilogue_warp_id[0]:
            cute.copy(tma_q, q_s[(None, buf)], q_g[(None, sub)])
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        barrier.arrive_and_wait()
        # K has no bias in Whisper.
        rOut_view.store(copy_r2s.retile(rK).load().to(gemm_kernel.c_dtype))
        buf = (base_seq + sub * 3 + 1) % gemm_kernel.num_c_stage
        cute.copy(copy_r2s, rOut_view, sC_view[(None, None, None, buf)])
        cute.arch.fence_proxy("async.shared", space="cta")
        barrier.arrive_and_wait()
        if warp_idx == gemm_kernel.epilogue_warp_id[0]:
            cute.copy(tma_k, k_s[(None, buf)], k_g[(None, sub)])
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        barrier.arrive_and_wait()
        # V bias uses the same exact two-rounding contract as Q.
        x = copy_r2s.retile(rV).load().to(gemm_kernel.c_dtype)
        bias = copy_r2s.retile(rbV).load()
        rOut_view.store((x + bias).to(gemm_kernel.c_dtype))
        buf = (base_seq + sub * 3 + 2) % gemm_kernel.num_c_stage
        cute.copy(copy_r2s, rOut_view, sC_view[(None, None, None, buf)])
        cute.arch.fence_proxy("async.shared", space="cta")
        barrier.arrive_and_wait()
        if warp_idx == gemm_kernel.epilogue_warp_id[0]:
            cute.copy(tma_v, v_s[(None, buf)], v_g[(None, sub)])
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        barrier.arrive_and_wait()
    barrier.arrive_and_wait()
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(state)
    state.advance()
    return state


class TripleQkvKernel(PersistentDenseGemmKernel):
    def _setup_attributes(self):
        tiled_mma = self._create_tiled_mma()
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (
            self.mma_tiler[0], self.mma_tiler[1], mma_inst_shape_k * 4
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1], self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        # Even for the direct-store epilogue, use the normal SM100 subtile
        # heuristic. A full 128x128 register tile would make each thread hold
        # both complete accumulator vectors and exhaust the register file.
        self.epi_tile = utils.sm100.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes()
        # Three N=128 FP32 accumulator regions fit TMEM only at one stage.
        # The allocator requires a power-of-two column count, so reserve the
        # full 512-column arena and leave the final 128 columns unused.
        self.num_acc_stage = 1

        a_one = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, 1
        )
        b_one = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, 1
        )
        bytes_per_stage = (
            cute.size_in_bytes(self.a_dtype, a_one)
            + 3 * cute.size_in_bytes(self.b_dtype, b_one)
        )
        c_one = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, 1
        )
        c_bytes_per_stage = cute.size_in_bytes(self.c_dtype, c_one)
        # Expose the two independent pipeline footprints as compile-time dials.
        # The incumbent auto-allocation resolves to AB=3/C=14 for 64x128 on
        # B200, while shallow variants intentionally leave enough SMEM for
        # a second resident CTA to hide TMA/L1TEX scoreboard latency.
        self.num_ab_stage = NUM_AB_STAGE
        self.num_c_stage = NUM_C_STAGE
        if self.num_ab_stage * bytes_per_stage + self.num_c_stage * c_bytes_per_stage + 1024 > self.smem_capacity:
            raise ValueError("requested A/B and epilogue stages exceed SMEM capacity")
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        self.c_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage
        )
        self.num_tmem_alloc_cols_one = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage, self.arch
        )
        required_tmem_cols = 3 * self.num_tmem_alloc_cols_one
        if required_tmem_cols > 512:
            raise ValueError("triple accumulator exceeds SM100's 512 TMEM columns")
        self.num_tmem_alloc_cols = 512

    @cute.jit
    def __call__(self, a, bq, bk, bv, bias_q, bias_v, q, k, v,
                 max_active_clusters: cutlass.Constexpr, stream: cuda.CUstream):
        self.a_dtype = a.element_type
        self.b_dtype = bq.element_type
        self.c_dtype = q.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(bq).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(q)
        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_a, desc_a = cute.nvgpu.make_tiled_tma_atom_A(a_op, a, a_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_bq, desc_bq = cute.nvgpu.make_tiled_tma_atom_B(b_op, bq, b_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        tma_bk, desc_bk = cute.nvgpu.make_tiled_tma_atom_B(b_op, bk, b_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        tma_bv, desc_bv = cute.nvgpu.make_tiled_tma_atom_B(b_op, bv, b_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        epi_smem = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_q, desc_q = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), q, epi_smem, self.epi_tile)
        tma_k, desc_k = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), k, epi_smem, self.epi_tile)
        tma_v, desc_v = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), v, epi_smem, self.epi_tile)
        self.num_tma_load_bytes = (cute.size_in_bytes(self.a_dtype, a_smem) + 3 * cute.size_in_bytes(self.b_dtype, b_smem)) * atom_thr_size
        sched, grid = self._compute_grid(q, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters)
        self.kernel(tiled_mma, tma_a, desc_a, tma_bq, desc_bq, tma_bk, desc_bk, tma_bv, desc_bv,
                    tma_q, desc_q, tma_k, desc_k, tma_v, desc_v, bias_q, bias_v,
                    self.cluster_layout_vmnk, self.a_smem_layout_staged, self.b_smem_layout_staged,
                    self.c_smem_layout_staged, self.epi_tile, sched, lambda x: x).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(*self.cluster_shape_mn, 1), stream=stream)

    @cute.kernel
    def kernel(self, tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor,
        tma_bq: cute.CopyAtom, mBQ_nkl: cute.Tensor,
        tma_bk: cute.CopyAtom, mBK_nkl: cute.Tensor,
        tma_bv: cute.CopyAtom, mBV_nkl: cute.Tensor,
        tma_q: Optional[cute.CopyAtom], mQ_mnl: cute.Tensor,
        tma_k: Optional[cute.CopyAtom], mK_mnl: cute.Tensor,
        tma_v: Optional[cute.CopyAtom], mV_mnl: cute.Tensor,
        bias_q: cute.Tensor, bias_v: cute.Tensor,
        cluster_layout_vmnk: cute.Layout, a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile, tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_bq)
            cpasync.prefetch_descriptor(tma_bk)
            cpasync.prefetch_descriptor(tma_bv)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_q)
                cpasync.prefetch_descriptor(tma_k)
                cpasync.prefetch_descriptor(tma_v)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        # Define shared storage for kernel
        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem_dealloc_barrier = None
        if cutlass.const_expr(not self.use_tma_store):
            tmem_dealloc_barrier = pipeline.NamedBarrier(
                barrier_id=self.tmem_dealloc_sync_bar_id,
                num_threads=32 * len(self.epilogue_warp_id),
            )
        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        #
        # Setup smem tensor A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sBQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sBK = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sBV = smem.allocate_tensor(
            element_type=self.b_dtype, layout=b_smem_layout_staged.outer,
            byte_alignment=128, swizzle=b_smem_layout_staged.inner)

        #
        # Compute multicast mask for A/B buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gBQ_nkl = cute.local_tile(mBQ_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gBK_nkl = cute.local_tile(mBK_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gBV_nkl = cute.local_tile(mBV_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gQ_mnl = cute.local_tile(mQ_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        gK_mnl = cute.local_tile(mK_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        gV_mnl = cute.local_tile(mV_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        gBQ_bias = cute.local_tile(bias_q, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        gBV_bias = cute.local_tile(bias_v, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgBQ = thr_mma.partition_B(gBQ_nkl)
        tCgBK = thr_mma.partition_B(gBK_nkl)
        tCgBV = thr_mma.partition_B(gBV_nkl)
        tCgQ = thr_mma.partition_C(gQ_mnl)
        tCgK = thr_mma.partition_C(gK_mnl)
        tCgV = thr_mma.partition_C(gV_mnl)
        tCgBQ_bias = thr_mma.partition_C(gBQ_bias)
        tCgBV_bias = thr_mma.partition_C(gBV_bias)


        #
        # Partition global/shared tensor for TMA load A/B
        #
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tBQsBQ, tBQgBQ = cpasync.tma_partition(tma_bq, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sBQ, 0, 3), cute.group_modes(tCgBQ, 0, 3))
        tBKsBK, tBKgBK = cpasync.tma_partition(tma_bk, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sBK, 0, 3), cute.group_modes(tCgBK, 0, 3))
        tBVsBV, tBVgBV = cpasync.tma_partition(tma_bv, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sBV, 0, 3), cute.group_modes(tCgBV, 0, 3))


        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrBQ = tiled_mma.make_fragment_B(sBQ)
        tCrBK = tiled_mma.make_fragment_B(sBK)
        tCrBV = tiled_mma.make_fragment_B(sBV)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        #
        # Construct the scheduler
        #
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp
        #

        if warp_idx == self.tma_warp_id:
            #
            # Persistent tile scheduling loop
            #

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), RestK)
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                # ((atom_v, rest_v), RestK)
                tBQgBQ_slice = tBQgBQ[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]
                tBKgBK_slice = tBKgBK[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]
                tBVgBV_slice = tBVgBV[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Tma load loop
                #
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)

                    # The scheduler is M-major: consecutive CTAs share each
                    # Q/K/V weight tile across all 12 M tiles. Keep those B
                    # sectors resident, while making the much less temporally
                    # local A stream first-to-evict. This changes no work or
                    # reduction order and preserves the packed Q/K/V reuse.
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                        # TMA copy's runtime field requires a DSL i64, not
                        # the similarly named Python CacheEvictionPriority enum.
                        cache_policy=cutlass.Int64(1),  # EVICT_FIRST
                    )
                    cute.copy(tma_bq, tBQgBQ_slice[(None, handle.count)],
                              tBQsBQ[(None, handle.index)], tma_bar_ptr=handle.barrier,
                              mcast_mask=b_full_mcast_mask,
                              cache_policy=cutlass.Int64(2))  # EVICT_LAST
                    cute.copy(tma_bk, tBKgBK_slice[(None, handle.count)],
                              tBKsBK[(None, handle.index)], tma_bar_ptr=handle.barrier,
                              mcast_mask=b_full_mcast_mask,
                              cache_policy=cutlass.Int64(2))  # EVICT_LAST
                    cute.copy(tma_bv, tBVgBV_slice[(None, handle.count)],
                              tBVsBV[(None, handle.index)], tma_bar_ptr=handle.barrier,
                              mcast_mask=b_full_mcast_mask,
                              cache_policy=cutlass.Int64(2))  # EVICT_LAST

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait A/B buffer empty
            #
            ab_producer.tail()

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAccQ_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAccK_base = cute.make_tensor(tmem_ptr + self.num_tmem_alloc_cols_one, tCtAcc_fake.layout)
            tCtAccV_base = cute.make_tensor(tmem_ptr + 2 * self.num_tmem_alloc_cols_one, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop
            #

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )

                # Set tensor memory buffer for current tile
                # (MMA, MMA_M, MMA_N)
                tCtAccQ = tCtAccQ_base[(None, None, None, acc_producer_state.index)]
                tCtAccK = tCtAccK_base[(None, None, None, acc_producer_state.index)]
                tCtAccV = tCtAccV_base[(None, None, None, acc_producer_state.index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                #
                # Reset the ACCUMULATE field for each tile
                #
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                #
                # Mma mainloop
                #
                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                        # tCtAcc += tCrA * tCrB
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblk_crd = (None, None, kblk_idx, handle.index)

                            cute.gemm(tiled_mma, tCtAccQ, tCrA[kblk_crd], tCrBQ[kblk_crd], tCtAccQ)
                            cute.gemm(tiled_mma, tCtAccK, tCrA[kblk_crd], tCrBK[kblk_crd], tCtAccK)
                            cute.gemm(tiled_mma, tCtAccV, tCrA[kblk_crd], tCrBV[kblk_crd], tCtAccV)
                            # All accumulator regions are now initialized.
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Async arrive AB buffer empty
                        handle.release()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                #
                # Async arrive accumulator buffer full
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        sC = None
        if cutlass.const_expr(self.use_tma_store):
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC = smem.allocate_tensor(
                element_type=self.c_dtype,
                layout=c_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=c_smem_layout_staged.inner,
            )

        #
        # Specialized epilogue warps
        #
        if warp_idx < self.mma_warp_id:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAccQ_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAccK_base = cute.make_tensor(tmem_ptr + self.num_tmem_alloc_cols_one, tCtAcc_fake.layout)
            tCtAccV_base = cute.make_tensor(tmem_ptr + 2 * self.num_tmem_alloc_cols_one, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop for epilogue
            #
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            if cutlass.const_expr(self.use_tma_store):
                assert tma_q is not None and tma_k is not None and tma_v is not None and sC is not None
                c_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    32 * len(self.epilogue_warp_id),
                )
                c_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.num_c_stage, producer_group=c_producer_group
                )
            while work_tile.is_valid_tile:
                # Get tile coord from tile scheduler
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                #
                # Pre-advance to next tile
                #
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                num_tiles_executed = tile_sched.num_tiles_executed
                if cutlass.const_expr(self.use_tma_store):
                    acc_consumer_state = triple_qkv_epilogue_tma(
                        self, tidx, warp_idx, tma_q, tma_k, tma_v,
                        tCtAccQ_base, tCtAccK_base, tCtAccV_base, sC,
                        tCgQ, tCgK, tCgV, tCgBQ_bias, tCgBV_bias,
                        epi_tile, num_tiles_executed, mma_tile_coord_mnl,
                        acc_consumer_state, acc_pipeline, c_pipeline)

                else:
                    acc_consumer_state = utils.gemm.sm100.epilogue(
                        self, tidx, tCtAccQ_base, tCgQ, epi_tile,
                        epilogue_op, mma_tile_coord_mnl,
                        acc_consumer_state, acc_pipeline,
                    )

            if cutlass.const_expr(self.use_tma_store):
                # Wait for C store complete
                c_pipeline.producer_tail()
            else:
                # Synchronize before TMEM dealloc (done by the caller)
                tmem_dealloc_barrier.arrive_and_wait()

            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


_triple = TripleQkvKernel(
    acc_dtype=cutlass.Float32, use_2cta_instrs=TWO_CTA,
    mma_tiler_mn=(TILE_M, TILE_N), cluster_shape_mn=(CLUSTER_M, CLUSTER_N),
    use_tma_store=True)

@cute.jit
def triple_qkv_entry(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor,
                     x: cute.Tensor, wq: cute.Tensor, bq: cute.Tensor,
                     wk: cute.Tensor, wv: cute.Tensor, bv: cute.Tensor,
                     stream: cuda.CUstream):
    bias_layout = cute.make_layout((M, N, 1), stride=(0, 1, 0))
    bq_view = cute.make_tensor(bq.iterator, bias_layout)
    bv_view = cute.make_tensor(bv.iterator, bias_layout)
    _triple(x, wq, wk, wv, bq_view, bv_view, q, k, v, MAX_CLUSTERS, stream)
