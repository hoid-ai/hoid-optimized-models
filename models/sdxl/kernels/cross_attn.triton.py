@triton.jit
def flash_attn_bshd_wide_k(out_ptr, q_ptr, kv_wide_ptr, scale_ptr, dyn_dims,
              Q_LEN: tl.constexpr, KV_LEN: tl.constexpr,
              HEAD_DIM: tl.constexpr, NUM_HEADS: tl.constexpr,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              KV_ROW_STRIDE: tl.constexpr, K_COL: tl.constexpr, V_COL: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // NUM_HEADS
    h = bh % NUM_HEADS
    q_base = (b * Q_LEN * NUM_HEADS + h) * HEAD_DIM
    kv_row_base = b * KV_LEN * KV_ROW_STRIDE
    k_base = kv_row_base + K_COL + h * HEAD_DIM
    v_base = kv_row_base + V_COL + h * HEAD_DIM
    q_desc = tl.make_tensor_descriptor(q_ptr + q_base, shape=[Q_LEN, HEAD_DIM],
        strides=[NUM_HEADS * HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    k_desc = tl.make_tensor_descriptor(kv_wide_ptr + k_base, shape=[KV_LEN, HEAD_DIM],
        strides=[KV_ROW_STRIDE, 1], block_shape=[BLOCK_N, HEAD_DIM])
    v_desc = tl.make_tensor_descriptor(kv_wide_ptr + v_base, shape=[KV_LEN, HEAD_DIM],
        strides=[KV_ROW_STRIDE, 1], block_shape=[BLOCK_N, HEAD_DIM])
    o_desc = tl.make_tensor_descriptor(out_ptr + q_base, shape=[Q_LEN, HEAD_DIM],
        strides=[NUM_HEADS * HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    q = q_desc.load([pid_m * BLOCK_M, 0])
    scale = tl.load(scale_ptr)
    row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    row_sum = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    for key0 in range(0, KV_LEN, BLOCK_N):
        k = tl.trans(k_desc.load([key0, 0]))
        scores = tl.dot(q, k, out_dtype=tl.float32).to(tl.bfloat16)
        scores = (scores * scale).to(tl.bfloat16)
        keys = key0 + tl.arange(0, BLOCK_N)
        scores = tl.where(keys[None, :] < KV_LEN, scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        alpha = tl.exp2((row_max - new_max) * 1.4426950408889634)
        probs = tl.exp2((scores - new_max[:, None]) * 1.4426950408889634)
        v = v_desc.load([key0, 0])
        acc = acc * alpha[:, None] + tl.dot(probs.to(tl.bfloat16), v, out_dtype=tl.float32)
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        row_max = new_max
    result = acc / row_sum[:, None]
    o_desc.store([pid_m * BLOCK_M, 0], result.to(tl.bfloat16))
