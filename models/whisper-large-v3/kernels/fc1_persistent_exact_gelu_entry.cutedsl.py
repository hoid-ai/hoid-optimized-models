# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
"""First Blackwell persistent dual-GEMM GEGLU prototype for hoid.

Derived from CUTLASS's dense_gemm_persistent.py.  The material differences are:
one A TMA stream, two B TMA streams, two disjoint TMEM accumulator regions,
and a direct-store epilogue that adds the two bias halves, evaluates the run's
poly8 GELU on the gate half, multiplies by the value half, and writes only O.
"""
import os
import sys
from typing import Optional, Tuple, Type, Union

import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.common import CacheEvictionPriority
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


def _poly8(x):
    """The exact coefficient set and branch points used by s1_pair.yaml."""
    zero = cute.full_like(x, 0.0)
    a = cute.where(x < zero, -x, x)
    y = cute.full_like(x, -1.0031614511e-04)
    y = y * a + cute.full_like(x, 2.2493980359e-03)
    y = y * a + cute.full_like(x, -2.0416898653e-02)
    y = y * a + cute.full_like(x, 9.3506507576e-02)
    y = y * a + cute.full_like(x, -2.0888346434e-01)
    y = y * a + cute.full_like(x, 1.1582269520e-01)
    y = y * a + cute.full_like(x, 3.5170763731e-01)
    y = y * a + cute.full_like(x, 5.0789552927e-01)
    y = y * a + cute.full_like(x, -3.0931478250e-04)
    central = cute.where(x < zero, y - a, y)
    outside = cute.where(x > zero, x, zero)
    return cute.where(a >= cute.full_like(x, 5.0), outside, central)


@cute.jit
def bias_exact_gelu_epilogue_tma(
    gemm_kernel,
    epi_tidx,
    warp_idx,
    tma_atom_c,
    tCtAcc_base,
    sC,
    tCgC_base,
    tCgBias_base,
    epi_tile,
    num_tiles_executed,
    mma_tile_coord_mnl,
    acc_consumer_state,
    acc_pipeline,
    c_pipeline,
):
    """Single-TMEM exact-GELU epilogue staged through SMEM and stored by TMA."""
    tCgC = transform_partitioned_tensor_layout(tCgC_base)
    tCgBias = transform_partitioned_tensor_layout(tCgBias_base)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )
    tTR_rOut = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tiled_copy_r2s, tRS_rOut, tRS_sC = epilogue_smem_copy_and_partition(
        gemm_kernel, tiled_copy_t2r, tTR_rOut, epi_tidx, sC
    )

    tCgC_epi = cute.flat_divide(tCgC, epi_tile)
    bSG_sC, bSG_gC_part = cpasync.tma_partition(
        tma_atom_c, 0, cute.make_layout(1),
        cute.group_modes(sC, 0, 2), cute.group_modes(tCgC_epi, 0, 2),
    )
    bSG_gC = bSG_gC_part[(None, None, None, *mma_tile_coord_mnl)]

    thr_copy = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_gBias_part = thr_copy.partition_D(cute.flat_divide(tCgBias, epi_tile))
    tTR_rBias = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tTR_gBias = tTR_gBias_part[(None, None, None, None, None, *mma_tile_coord_mnl)]

    epilog_sync_barrier = pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id,
        num_threads=WARP_SIZE * len(gemm_kernel.epilogue_warp_id),
    )
    tTR_tAcc = tTR_tAcc_base[
        (None, None, None, None, None, acc_consumer_state.index)
    ]
    acc_pipeline.consumer_wait(acc_consumer_state)
    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    tTR_gBias = cute.group_modes(tTR_gBias, 3, cute.rank(tTR_gBias))
    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
    num_prev_subtiles = num_tiles_executed * subtile_cnt
    for subtile_idx in range(subtile_cnt):
        cute.copy(
            tiled_copy_t2r,
            tTR_tAcc[(None, None, None, subtile_idx)],
            tTR_rAcc,
        )
        cute.autovec_copy(tTR_gBias[(None, None, None, subtile_idx)], tTR_rBias)

        # Preserve the incumbent's two bf16 rounding boundaries: GEMM landing,
        # then bias addition. GELU itself evaluates in fp32 with CUDA 13 erff's
        # exact inlined approximation and rounds only at the final store.
        x = tiled_copy_r2s.retile(tTR_rAcc).load().to(gemm_kernel.c_dtype)
        bias = tiled_copy_r2s.retile(tTR_rBias).load()
        x = (x + bias).to(gemm_kernel.c_dtype).to(gemm_kernel.acc_dtype)
        a = x * cute.full_like(x, 0.7071067690849304)
        ax = cute.absf(a)
        threshold = cute.full_like(a, 1.002959966659546)
        large = ax >= threshold
        z = cute.where(large, ax, a * a)
        p = cute.where(large,
            cute.full_like(a, 0.00011219871521461755),
            cute.full_like(a, 8.483494457323104e-05))
        p = cute.fma(p, z, cute.where(large,
            cute.full_like(a, -0.0013275252422317863),
            cute.full_like(a, -0.0008213091641664505)))
        p = cute.fma(p, z, cute.where(large,
            cute.full_like(a, 0.00839653518050909),
            cute.full_like(a, 0.005213488824665546)))
        p = cute.fma(p, z, cute.where(large,
            cute.full_like(a, -0.04024658352136612),
            cute.full_like(a, -0.026868773624300957)))
        p = cute.fma(p, z, cute.where(large,
            cute.full_like(a, 0.15950430929660797),
            cute.full_like(a, 0.11284004896879196)))
        p = cute.fma(p, z, cute.where(large,
            cute.full_like(a, 0.9129176735877991),
            cute.full_like(a, -0.37612664699554443)))
        p = cute.fma(p, z, cute.where(large,
            cute.full_like(a, 0.6290600299835205),
            cute.full_like(a, 0.12837915122509003)))
        t = cute.where(large, -z, a)
        erf_small_or_log = cute.fma(p, t, t)
        outer = cute.copysign(
            cute.full_like(a, 1.0) - cute.exp2(erf_small_or_log, fastmath=True), a
        )
        erf_x = cute.where(ax < threshold, erf_small_or_log, outer)
        y = (x * cute.full_like(x, 0.5)) * (cute.full_like(x, 1.0) + erf_x)
        tRS_rOut.store(y.to(gemm_kernel.c_dtype))

        c_buffer = (num_prev_subtiles + subtile_idx) % gemm_kernel.num_c_stage
        cute.copy(tiled_copy_r2s, tRS_rOut, tRS_sC[(None, None, None, c_buffer)])
        cute.arch.fence_proxy("async.shared", space="cta")
        epilog_sync_barrier.arrive_and_wait()
        if warp_idx == gemm_kernel.epilogue_warp_id[0]:
            cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)], bSG_gC[(None, subtile_idx)])
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        epilog_sync_barrier.arrive_and_wait()

    epilog_sync_barrier.arrive_and_wait()
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()
    return acc_consumer_state

class BiasExactGeluKernel(PersistentDenseGemmKernel):
    @cute.jit
    def __call__(self, a, b, bias, c,
                 max_active_clusters: cutlass.Constexpr, stream: cuda.CUstream):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)
        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_a, desc_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_b, desc_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        epi_smem = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_c, desc_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem, self.epi_tile
        )
        self.num_tma_load_bytes = (
            cute.size_in_bytes(self.a_dtype, a_smem)
            + cute.size_in_bytes(self.b_dtype, b_smem)
        ) * atom_thr_size
        sched, grid = self._compute_grid(
            c, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )
        self.kernel(
            tiled_mma, tma_a, desc_a, tma_b, desc_b, tma_c, desc_c, bias,
            self.cluster_layout_vmnk, self.a_smem_layout_staged,
            self.b_smem_layout_staged, self.c_smem_layout_staged,
            self.epi_tile, sched, lambda x: x,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: Optional[cute.CopyAtom],
        mC_mnl: cute.Tensor,
        mBias_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
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
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(self.use_tma_store):
                cpasync.prefetch_descriptor(tma_atom_c)

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
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

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
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        gBias_mnl = cute.local_tile(
            mBias_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)
        tCgBias = thr_mma.partition_C(gBias_mnl)

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
        tBsB, tBgBV = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
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
                tBgBV_slice = tBgBV[
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

                    # TMA load A/B
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgBV_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

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
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

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
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

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

                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblk_crd],
                                tCrB[kblk_crd],
                                tCtAcc,
                            )
                            # The accumulator region is now initialized.
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
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Persistent tile scheduling loop for epilogue
            #
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            if cutlass.const_expr(self.use_tma_store):
                assert tma_atom_c is not None and sC is not None
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
                    acc_consumer_state = bias_exact_gelu_epilogue_tma(
                        self,
                        tidx,
                        warp_idx,
                        tma_atom_c,
                        tCtAcc_base,
                        sC,
                        tCgC,
                        tCgBias,
                        epi_tile,
                        num_tiles_executed,
                        mma_tile_coord_mnl,
                        acc_consumer_state,
                        acc_pipeline,
                        c_pipeline,
                    )
                else:
                    acc_consumer_state = utils.gemm.sm100.epilogue(
                        self, tidx, tCtAcc_base, tCgC, epi_tile,
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



# CuTeDSL currently lacks a public fma binding in this pinned tree; this is the
# same shim used by the accepted CUDA-13 exact-erf EFC implementation.
from cutlass._mlir_helpers.math import fma as _cute_fma
cute.fma = _cute_fma

_gemm = BiasExactGeluKernel(
    acc_dtype=cutlass.Float32,
    use_2cta_instrs=TWO_CTA,
    mma_tiler_mn=(TILE_M, TILE_N),
    cluster_shape_mn=(CLUSTER_M, CLUSTER_N),
    use_tma_store=True,
)

@cute.jit
def fc1_persistent_exact_gelu_entry(d: cute.Tensor, x: cute.Tensor,
                                    w: cute.Tensor, bias: cute.Tensor,
                                    stream: cuda.CUstream):
    bias_layout = cute.make_layout((M, N, 1), stride=(0, 1, 0))
    bias_mn = cute.make_tensor(bias.iterator, bias_layout)
    _gemm(x, w, bias_mn, d, MAX_CLUSTERS, stream)
