"""GENERATED manifest of the winning hoid kernels — do not edit by hand.

Keyed by role. `file` names a source under kernels/ that is byte-identical
to the shipped artifact (`sha256` is over that source); `launch` is the
geometry hoid launched it with; the Triton rows carry hoid's measured
compile constants.
"""

KERNELS = {
    'embedding_add_ln': {
        'entry': 'bge_embedding_add_ln_warp_k',
        'format': 'cuda',
        'file': 'bge_embedding_add_ln_warp_k.cu',
        'sha256': '59ab8bde2cdd2d245b0f536cd43ebb881191ce5739e455771fb1600f3e2965a2',
        'count': 1,
        'launch': {'grid': ['4096', '1', '1'], 'block': ['256', '1', '1'], 'shared_mem': '0'},
    },
    'fa2': {
        'entry': 'bge_fa2_attention_nokb_postvb_late',
        'format': 'ptx',
        'file': 'bge_fa2_attention_nokb_postvb_late.ptx',
        'sha256': '506bbe50b5795db4ceafcc71423a0964ce454c0fb40afdfd02e9bd3eb734e95c',
        'count': 24,
        'launch': {'grid': ['4', '1024', '1'], 'block': ['128', '1', '1'], 'shared_mem': '49184'},
        'scratch': {'global_size': 0, 'global_align': 1, 'profile_size': 0},
    },
    'bias_residual_ln': {
        'entry': 'bge_bias_residual_ln_warp_packed_k',
        'format': 'cuda',
        'file': 'bge_bias_residual_ln_warp_packed_k.cu',
        'sha256': '959a14ff5002642d5d5599d408431f75ef12e5cc5d7705445eba884b1913a975',
        'count': 48,
        'launch': {'grid': ['4096', '1', '1'], 'block': ['256', '1', '1'], 'shared_mem': '0'},
    },
    'bias_gelu_lut': {
        'entry': 'bge_bias_gelu_bf16_lut_direct_k',
        'format': 'cuda',
        'file': 'bge_bias_gelu_bf16_lut_direct_k.cu',
        'sha256': '056479e9d5aa1772713655a10c21bae988a22d62dfd1c4ed39b402ccaa39a3f2',
        'count': 24,
        'launch': {'grid': ['16384', '1', '1'], 'block': ['256', '1', '1'], 'shared_mem': '0'},
    },
    'pool': {
        'entry': 'bge_pool_k',
        'format': 'cuda',
        'file': 'bge_pool_k.cu',
        'sha256': '2867ed07dba642af2092b0921abe5850be7969592960ff497ec1ce84d8158e71',
        'count': 1,
        'launch': {'grid': ['64', '1', '1'], 'block': ['256', '1', '1'], 'shared_mem': '0'},
    },
}
