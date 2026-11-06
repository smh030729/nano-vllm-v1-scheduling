from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def load_fused_qk_norm_rope():
    """Build/load dispatcher registrations before model warmup or graph capture."""
    from torch.utils.cpp_extension import load

    source = Path(__file__).resolve().parent / 'csrc'
    load(
        name='nanovllm_qk_norm_rope',
        sources=[str(source / 'qk_norm_rope.cpp'), str(source / 'qk_norm_rope.cu')],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3'],
        is_python_module=False,
    )
