# SPDX-License-Identifier: Apache-2.0
"""HydraGen attention
    github: https://github.com/ScalingIntelligence/hydragen/blob/main/hydragen/attention.py
"""

from typing import Optional

import torch

from vllm.attention.utils.fa_utils import (
    is_flash_attn_varlen_func_available,
)

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import flash_attn_varlen_func


def hydragen_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    fa_version: int,
    prefix_scheduler_metadata: Optional[torch.Tensor] = None,
    suffix_scheduler_metadata: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    # print("=== hydragen query shape", cu_query_lens.shape)
    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0

    # Prefix pass 
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=sliding_window,
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=prefix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
    )

    # Suffix pass
    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=sliding_window,
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        scheduler_metadata=suffix_scheduler_metadata,
        fa_version=fa_version,
        q_descale=q_descale.expand(descale_shape) if q_descale is not None else None,
        k_descale=k_descale.expand(descale_shape) if k_descale is not None else None,
        v_descale=v_descale.expand(descale_shape) if v_descale is not None else None,
    )

    # combine lse
    # O1 = softmax(A, dim=kv) V1 = exp(A − lse1) V1
    # O2 = softmax(B, dim=kv) V2 = exp(B − lse2) V2
    # O = exp(lse1 − lse) O1 + exp(lse2 − lse) O2

    lse_total = torch.logaddexp(prefix_lse, suffix_lse)
    w_prefix = torch.exp(prefix_lse - lse_total).unsqueeze(-1)
    w_suffix = torch.exp(suffix_lse - lse_total).unsqueeze(-1)

    prefix_out_htd = prefix_output.transpose(0, 1)
    suffix_out_htd = suffix_output.transpose(0, 1)
    merged_htd = w_prefix * prefix_out_htd + w_suffix * suffix_out_htd
    merged_thd = merged_htd.transpose(0, 1)

    output.copy_(merged_thd)
    return output


__all__ = ["hydragen_attention"]


