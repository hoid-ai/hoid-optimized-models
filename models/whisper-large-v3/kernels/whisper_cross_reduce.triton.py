@triton.jit
def whisper_cross_reduce(out_ptr, partial_ptr, dyn_dims, HEAD_DIM: tl.constexpr, SPLITS: tl.constexpr):
    head = tl.program_id(0)
    splits = tl.arange(0, SPLITS)
    cols = tl.arange(0, HEAD_DIM)
    base = (head * SPLITS + splits) * (HEAD_DIM + 2)
    m = tl.load(partial_ptr + base)
    l = tl.load(partial_ptr + base + 1)
    gm = tl.max(m, axis=0)
    scale = tl.exp(m - gm)
    denom = tl.sum(l * scale, axis=0)
    acc = tl.load(partial_ptr + base[:, None] + 2 + cols[None, :])
    result = tl.sum(acc * scale[:, None], axis=0) / denom
    tl.store(out_ptr + head * HEAD_DIM + cols, result)
