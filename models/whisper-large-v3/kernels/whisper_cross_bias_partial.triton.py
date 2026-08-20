@triton.jit
def whisper_cross_bias_partial(out_ptr, qraw_ptr, bias_ptr, k_ptr, v_ptr, dyn_dims, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, SPLITS: tl.constexpr, CHUNK: tl.constexpr):
    STRIDE = 1280
    SK = 1500
    SCALE = 0.125
    pid = tl.program_id(0)
    head = pid // SPLITS
    split = pid - head * SPLITS
    cols = tl.arange(0, HEAD_DIM)
    keys = split * CHUNK + tl.arange(0, BLOCK_N)
    mask = (keys < SK) & (keys < (split + 1) * CHUNK)
    q = (tl.load(qraw_ptr + head * HEAD_DIM + cols).to(tl.float32) + tl.load(bias_ptr + head * HEAD_DIM + cols).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    k = tl.load(k_ptr + keys[:, None] * STRIDE + head * HEAD_DIM + cols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    scores = tl.sum(k * q[None, :], axis=1) * SCALE
    scores = tl.where(mask, scores, -float("inf"))
    m = tl.max(scores, axis=0)
    p = tl.exp(scores - m)
    p = tl.where(mask, p, 0.0)
    l = tl.sum(p, axis=0)
    v = tl.load(v_ptr + keys[:, None] * STRIDE + head * HEAD_DIM + cols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    acc = tl.sum(p[:, None] * v, axis=0)
    base = pid * (HEAD_DIM + 2)
    tl.store(out_ptr + base, m)
    tl.store(out_ptr + base + 1, l)
    tl.store(out_ptr + base + 2 + cols, acc)
