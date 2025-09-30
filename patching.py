from transformers.models.gpt_oss import modeling_gpt_oss
from attention_kernel import triton_flash_attention

_original_eager_attention_forward = modeling_gpt_oss.eager_attention_forward

def patch_flash_attention():
    modeling_gpt_oss.eager_attention_forward = triton_flash_attention

def unpatch_flash_attention():
    modeling_gpt_oss.eager_attention_forward = _original_eager_attention_forward