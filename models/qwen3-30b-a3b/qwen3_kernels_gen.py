"""GENERATED manifest of the hoid-emitted kernels under kernels/ — do not
edit. Custom kernels are byte-identical to the winning stack's (the
fullcta_add_rmsnorm_<id>_k family canonicalized to one entry name);
prepare_qkv/rmsnorm are hoid's generated builtin sources. Loaders re-hash
kernels/<file> against sha256 at import.
"""

KERNELS = {
    'attn_combine_k': {
        'format': 'cuda',
        'file': 'attn_combine_k.cu',
        'sha256': '747d98996a8e905690d4206890e0a37b0c260454b1444e129a8da3044bd4ca78',
        'count': 48,
        'launch': {
            'grid': ['seq:dyn', '32', '1'],
            'block': ['128', '1', '1'],
            'shared_mem': '0',
        },
    },
    'attn_gqa8_subsplit64_partial_k': {
        'format': 'cuda',
        'file': 'attn_gqa8_subsplit64_partial_k.cu',
        'sha256': 'c53650a75a3ad1141ee7b5507d60a4a3a3531368cff3b031f6ac740126d591c8',
        'count': 48,
        'launch': {
            'grid': ['seq:dyn', '4', '64'],
            'block': ['256', '1', '1'],
            'shared_mem': '0',
        },
    },
    'attn_subsplit64_pair_reduce_k': {
        'format': 'cuda',
        'file': 'attn_subsplit64_pair_reduce_k.cu',
        'sha256': 'e39ac6c44f345556a85c4859dafcc649833bfa35bec28776a1c619875b989398',
        'count': 48,
        'launch': {
            'grid': ['seq:dyn', '32', '32'],
            'block': ['128', '1', '1'],
            'shared_mem': '0',
        },
    },
    'attn_subsplit_reduce_k': {
        'format': 'cuda',
        'file': 'attn_subsplit_reduce_k.cu',
        'sha256': '7d9706efbb14089f38cca0c9ced2066cbbe796f7b54038c2a58bf4bf3d770780',
        'count': 48,
        'launch': {
            'grid': ['seq:dyn', '32', '8'],
            'block': ['128', '1', '1'],
            'shared_mem': '0',
        },
    },
    'fullcta_add_rmsnorm_final_k': {
        'format': 'cuda',
        'file': 'fullcta_add_rmsnorm_final_k.cu',
        'sha256': '7784a46b7fd0cdd6df10593dd6801133bab6e3133f738f95d45d8019ee746f96',
        'count': 1,
        'launch': {
            'grid': ['seq:dyn', '1', '1'],
            'block': ['1024', '1', '1'],
            'shared_mem': '0',
        },
    },
    'fullcta_add_rmsnorm_k': {
        'format': 'cuda',
        'file': 'fullcta_add_rmsnorm_k.cu',
        'sha256': 'd1e83cbad40bc14794473e2c6e244f818be17a09947f44beb3eddbc183638247',
        'count': 95,
        'launch': {
            'grid': ['seq:dyn', '1', '1'],
            'block': ['1024', '1', '1'],
            'shared_mem': '0',
        },
    },
    'moe_down_accum_k': {
        'format': 'cuda',
        'file': 'moe_down_accum_k.cu',
        'sha256': 'f88b381144f617b7f36f0d8ce3852d5064352b32fd42e30399ccca5580d8f495',
        'count': 48,
        'launch': {
            'grid': ['seq:dyn', '256', '1'],
            'block': ['256', '1', '1'],
            'shared_mem': '24608',
        },
    },
    'moe_gate_up_silu_k': {
        'format': 'cuda',
        'file': 'moe_gate_up_silu_k.cu',
        'sha256': '81315e05c891e10ee4bcd411a43a802d560b4e24f9d6caa1e16ecb1f66449bf9',
        'count': 48,
        'launch': {
            'grid': ['seq:dyn', '8', '96'],
            'block': ['256', '1', '1'],
            'shared_mem': '8192',
        },
    },
    'moe_router_parallel_topk_k': {
        'format': 'cuda',
        'file': 'moe_router_parallel_topk_k.cu',
        'sha256': 'cab8ec2f8b1cbb915e4cba40f9ba7a3d007517d05d171c6af0a226122128a491',
        'count': 48,
        'launch': {
            'grid': ['seq:dyn', '1', '1'],
            'block': ['128', '1', '1'],
            'shared_mem': '0',
        },
    },
    'prepare_qkv_k': {
        'format': 'cuda',
        'file': 'prepare_qkv_k.cu',
        'sha256': '3317dabae1082477b586505427041a735ec1e54a034940d6151c52d7421a51c8',
        'count': 48,
        'builtin_op': 'prepare_qkv',
        'launch': {
            'grid': ['seq:dyn * 40', '1', '1'],
            'block': ['32', '1', '1'],
            'shared_mem': '0',
        },
    },
    'rmsnorm_k': {
        'format': 'cuda',
        'file': 'rmsnorm_k.cu',
        'sha256': '8eb775a9bd5be9039aff75ef823af3a157370a56690a37248584ad756eb54e5b',
        'count': 1,
        'builtin_op': 'rmsnorm',
        'launch': {
            'grid': ['seq:dyn', '1', '1'],
            'block': ['256', '1', '1'],
            'shared_mem': '0',
        },
    },
}
