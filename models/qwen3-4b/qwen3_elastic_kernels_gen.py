# The elastic prefill attention, byte-identical to the llama31 port's. The
# kernel is model-agnostic and this model shares its full attention geometry
# (NQH=32, GROUP=4, HDIM=128, 8 KV heads, head-major cache rows);
# MAXSEQ is re-baked per engine build like the CUDA kernels.
KERNELS = {'fmha_prefill_elastic': {'entry': 'fmha_prefill_elastic', 'format': 'triton', 'file': 'kernels/fmha_prefill_elastic.triton.py', 'count': 32, 'launch': {'grid': ['ceil_div(seq:dyn, 128)', '32', '1']}, 'params': {'num_warps': 4, 'num_stages': 2, 'constexprs': {'BLOCK_M': 128, 'BLOCK_N': 64, 'GROUP': 4, 'HDIM': 128, 'MAXSEQ': 9216, 'NQH': 32}, 'signature': {'out': '*bf16', 'q': '*bf16', 'kc': '*bf16', 'vc': '*bf16', 'dyn_dims': '*i32'}}, 'outputs': [{'dtype': 'bf16', 'shape': {'dims': ['seq:dyn', '32', '128'], 'strides': ['1 * 128 * 32', '1 * 128', '1']}}], 'n_inputs': 3, 'sha256': '9323e16d37a5c915a49d3a278151c89f142ee73bdddf6766aad0db18f839104f'}}
