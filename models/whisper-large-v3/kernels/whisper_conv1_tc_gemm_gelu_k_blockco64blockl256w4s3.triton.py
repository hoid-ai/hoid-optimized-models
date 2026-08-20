# Tensor-core implicit-GEMM stride-1 width-3 conv + bias + exact erf GELU for
# the fixed Whisper conv2 shape (X [128,3000] -> out [1280,3000]).
@triton.jit
def whisper_conv1_tc_gemm_gelu_k_blockco64blockl256w4s3(out_ptr, x_ptr, w_ptr, b_ptr, dyn_dims, BLOCK_CO: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr):
            OC = 1280
            IC = 128
            LIN = 3000
            LOUT = 3000
            pid_l = tl.program_id(0)
            pid_co = tl.program_id(1)
            co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
            ls = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
            acc = tl.zeros((BLOCK_CO, BLOCK_L), tl.float32)
            for t in tl.static_range(3):
                for ci0 in range(0, IC, BLOCK_K):
                    ci = ci0 + tl.arange(0, BLOCK_K)
                    w = tl.load(w_ptr + co[:, None] * (IC * 3) + ci[None, :] * 3 + t,
                                mask=co[:, None] < OC, other=0.0)
                    src = ls + t - 1
                    x = tl.load(x_ptr + ci[:, None] * LIN + src[None, :],
                                mask=(src[None, :] >= 0) & (src[None, :] < LIN) & (ls[None, :] < LOUT),
                                other=0.0)
                    acc += tl.dot(w, x, out_dtype=tl.float32)
            bias = tl.load(b_ptr + co, mask=co < OC, other=0.0).to(tl.float32)
            y = acc + bias[:, None]
            y = y.to(tl.bfloat16).to(tl.float32)
            g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
            tl.store(out_ptr + co[:, None] * LOUT + ls[None, :], g,
                     mask=(co[:, None] < OC) & (ls[None, :] < LOUT))
