@triton.jit
def fmha_prefill_elastic(out, q, kc, vc, dyn_dims,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         HDIM: tl.constexpr, NQH: tl.constexpr,
                         GROUP: tl.constexpr, MAXSEQ: tl.constexpr):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.load(dyn_dims + 18)
    kvh = h // GROUP
    SCALE_LOG2E: tl.constexpr = 0.12754617340254743  # 128^-0.5 * log2(e)
    NKVH: tl.constexpr = 8
    desc_k = tl.make_tensor_descriptor(kc, shape=[NKVH * MAXSEQ, HDIM],
                                       strides=[HDIM, 1],
                                       block_shape=[BLOCK_N, HDIM])
    desc_v = tl.make_tensor_descriptor(vc, shape=[NKVH * MAXSEQ, HDIM],
                                       strides=[HDIM, 1],
                                       block_shape=[BLOCK_N, HDIM])
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HDIM)
    qm = offs_m < s
    qb = tl.load(q + (offs_m[:, None] * NQH + h) * HDIM + offs_d[None, :],
                 mask=qm[:, None], other=0.0)
    acc = tl.zeros((BLOCK_M, HDIM), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    base = kvh * MAXSEQ
    # phase 1: tiles strictly below the diagonal block — no causal mask needed
    lo = pid_m * BLOCK_M
    for start_n in range(0, lo, BLOCK_N):
        kb = desc_k.load([base + start_n, 0])
        scores = tl.dot(qb, tl.trans(kb)) * SCALE_LOG2E
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(scores - m_new[:, None])
        acc = acc * alpha[:, None]
        vb = desc_v.load([base + start_n, 0])
        acc += tl.dot(p.to(tl.bfloat16), vb)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    # phase 2: the diagonal block — causal mask (covers the s bound for valid rows;
    # garbage keys past s only reach rows the store masks out)
    for start_n in range(lo, (pid_m + 1) * BLOCK_M, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kb = desc_k.load([base + start_n, 0])
        scores = tl.dot(qb, tl.trans(kb)) * SCALE_LOG2E
        causal = offs_m[:, None] >= offs_n[None, :]
        scores = tl.where(causal, scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(scores - m_new[:, None])
        acc = acc * alpha[:, None]
        vb = desc_v.load([base + start_n, 0])
        acc += tl.dot(p.to(tl.bfloat16), vb)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(out + (offs_m[:, None] * NQH + h) * HDIM + offs_d[None, :],
             acc.to(tl.bfloat16), mask=qm[:, None])
