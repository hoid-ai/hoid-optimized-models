@triton.jit
def whisper_self_bias_partial(out_ptr, qraw_ptr, qbias_ptr, knew_ptr, vraw_ptr, vbias_ptr, pos_ptr, kcache_ptr, vcache_ptr, dyn_dims, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr, SPLITS: tl.constexpr, CHUNK: tl.constexpr):
    MAX_SEQ = 448
    SCALE = 0.125
    pid = tl.program_id(0)
    head = pid // SPLITS
    split = pid - head * SPLITS
    cols = tl.arange(0, HEAD_DIM)
    pos = tl.load(pos_ptr)
    pos = tl.maximum(0, tl.minimum(pos, MAX_SEQ - 1))
    knew = tl.load(knew_ptr + head * HEAD_DIM + cols)
    vnew = (tl.load(vraw_ptr + head * HEAD_DIM + cols).to(tl.float32) + tl.load(vbias_ptr + head * HEAD_DIM + cols).to(tl.float32)).to(tl.bfloat16)
    cache_row = (head * MAX_SEQ + pos) * HEAD_DIM + cols
    tl.store(kcache_ptr + cache_row, knew)
    tl.store(vcache_ptr + cache_row, vnew)
    keys = split * CHUNK + tl.arange(0, BLOCK_N)
    mask = (keys < MAX_SEQ) & (keys < (split + 1) * CHUNK) & (keys <= pos)
    q = (tl.load(qraw_ptr + head * HEAD_DIM + cols).to(tl.float32) + tl.load(qbias_ptr + head * HEAD_DIM + cols).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    k = tl.load(kcache_ptr + (head * MAX_SEQ + keys[:, None]) * HEAD_DIM + cols[None, :], mask=mask[:, None], other=0.0)
    k = tl.where((keys[:, None] == pos) & mask[:, None], knew[None, :], k).to(tl.float32)
    scores = tl.sum(k * q[None, :], axis=1) * SCALE
    scores = tl.where(mask, scores, -float("inf"))
    m = tl.max(scores, axis=0)
    p = tl.where(mask, tl.exp(scores - m), 0.0)
    l = tl.sum(p, axis=0)
    v = tl.load(vcache_ptr + (head * MAX_SEQ + keys[:, None]) * HEAD_DIM + cols[None, :], mask=mask[:, None], other=0.0)
    v = tl.where((keys[:, None] == pos) & mask[:, None], vnew[None, :], v).to(tl.float32)
    acc = tl.sum(p[:, None] * v, axis=0)
    base = pid * (HEAD_DIM + 2)
    tl.store(out_ptr + base, m)
    tl.store(out_ptr + base + 1, l)
    tl.store(out_ptr + base + 2 + cols, acc)
