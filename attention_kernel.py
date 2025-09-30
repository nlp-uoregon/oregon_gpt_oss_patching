
import torch
import torch.nn.functional as F

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from typing import Optional

@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    Sinks,
    sm_scale,
    M,
    Out,  #
    Start_q,
    Z,
    H,
    KV_H,
    N_Q_CTX,
    N_KV_CTX,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    BANDWIDTH: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_kv_h = off_h // (H // KV_H) 

    # load attention sinks
    if Sinks is not None:
        sink = tl.load(Sinks + off_h).to(tl.float32) # sinks are shared across query heads
    else:
        sink = 0
    
    if Start_q is not None:
        start_q = tl.load(Start_q).to(tl.int32)
    else:
        start_q = 0

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], sink, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    q = Q.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])

    if BANDWIDTH:
        lo, hi = tl.maximum(0, start_q + start_m * BLOCK_M - BANDWIDTH), start_q + (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, start_q + (start_m + 1) * BLOCK_M

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        mask = (start_n + offs_n)[None, :] > (start_q + offs_m)[:, None]

        if BANDWIDTH:
            too_old = (start_n + offs_n[None, :]) < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | too_old

        k = K.load([off_z, off_kv_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM]).T
        qk = tl.dot(q, k, allow_tf32=False)

        qk = qk * qk_scale + tl.where(mask, -1.0e6, 0.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp(qk)
        alpha = tl.math.exp(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = V.load([off_z, off_kv_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        v = v.to(tl.float32)
        acc = tl.dot(p, v, acc, allow_tf32=False)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    sink = tl.math.exp(sink - m_i)
    z = l_i + sink
    acc = acc / z[:, None]
    m_i += tl.math.log(z)
    m_ptrs = M + off_hz * N_Q_CTX + offs_m
    tl.store(m_ptrs, m_i)
    acc = acc.to(Out.dtype)[None, None, :, :]
    Out.store([off_z, off_h, start_m * BLOCK_M, 0], acc)

@triton.jit
def _attn_bwd_precompute_D(
    D,
    DO,
    O,
    H,
    N_Q_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    o_i = O.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])
    do_i = DO.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])
    d_i = tl.sum(do_i.to(tl.float32) * o_i.to(tl.float32), axis=1)[None, None, :]
    D.store([off_z, off_h, start_m * BLOCK_M], d_i.to(D.dtype))

@triton.jit
def _attn_bwd(
    Q, K, V,
    Sinks,
    sm_scale,
    DO,
    DQ, DK, DV,
    Dsinks,
    M,
    D,
    Start_q,
    Z,
    H,
    KV_H,
    N_Q_CTX,
    N_KV_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BANDWIDTH: tl.constexpr,    
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    if Start_q is not None:
        start_q = tl.load(Start_q).to(tl.int32)
    else:
        start_q = 0
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    off_kv_h = off_h // (H // KV_H)
    

    if Sinks is not None:
        sink = tl.load(Sinks + off_h).to(tl.float32)
    else:
        sink = 0
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load q, do, m, and D
    q = Q.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])
    do = DO.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])
    m_block = tl.load(M + off_hz * N_Q_CTX + offs_m)
    D_block = tl.load(D + off_hz * N_Q_CTX + offs_m)
    
    # Initialize dq
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    
    # Compute dsinks
    p_sink = tl.math.exp(sink - m_block)
    d_sink = -p_sink * D_block
    d_sink = tl.sum(d_sink, axis=0)
    tl.atomic_add(Dsinks + off_h, d_sink, sem='relaxed') # no ordering required
    
    # Determine iteration range
    if BANDWIDTH:
        lo, hi = tl.maximum(0, start_q + start_m * BLOCK_M - BANDWIDTH), start_q + (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, start_q + (start_m + 1) * BLOCK_M
        
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        k = K.load([off_z, off_kv_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        v = V.load([off_z, off_kv_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        
        qk = tl.dot(q, k.T, allow_tf32=False) * sm_scale
        
        # causal mask
        mask = (start_n + offs_n)[None, :] > (start_q + offs_m)[:, None]
        if BANDWIDTH:
            window_mask = (start_n + offs_n[None, :]) < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | window_mask
        
        qk = qk + tl.where(mask, -1.0e6, 0.0)
        p = tl.math.exp(qk - m_block[:, None])
        
        dv_block = tl.dot(p.to(do.dtype).T, do, allow_tf32=False)
        dv_ptrs = DV + off_z * KV_H * N_KV_CTX * HEAD_DIM + off_kv_h * N_KV_CTX * HEAD_DIM + \
                (start_n + offs_n[:, None]) * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        tl.atomic_add(dv_ptrs, dv_block, sem='relaxed')
        
        dp = tl.dot(do, v.T, allow_tf32=False)
        ds = p * (dp - D_block[:, None]) 
        
        dk_block = tl.dot(ds.to(q.dtype).T, q, allow_tf32=False)
        dk_ptrs = DK + off_z * KV_H * N_KV_CTX * HEAD_DIM + off_kv_h * N_KV_CTX * HEAD_DIM + \
                (start_n + offs_n[:, None]) * HEAD_DIM + tl.arange(0, HEAD_DIM)[None, :]
        tl.atomic_add(dk_ptrs, dk_block*sm_scale, sem='relaxed')
        
        dq += tl.dot(ds.to(k.dtype), k, allow_tf32=False) * sm_scale
        
    dq = dq.to(Q.dtype)[None, None, :, :]
    DQ.store([off_z, off_h, start_m * BLOCK_M, 0], dq)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, sm_scale, bandwidth, start_q):
        bs, n_heads, n_ctx, HEAD_DIM_Q = q.shape
        bs, n_kv_heads, n_kv_ctx, HEAD_DIM_K = k.shape
        bs, n_kv_heads, n_kv_ctx, HEAD_DIM_V = v.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        BLOCK_M = 64
        BLOCK_N = 64
        # Pad q to multiple of BLOCK_M
        m_pad_size = BLOCK_M - n_ctx % BLOCK_M if n_ctx % BLOCK_M != 0 else 0
        q = torch.nn.functional.pad(q, (0, 0, 0, m_pad_size))
        
        # Pad k and v to multiple of BLOCK_N
        n_pad_size = BLOCK_N - n_kv_ctx % BLOCK_N if n_kv_ctx % BLOCK_N != 0 else 0
        k = torch.nn.functional.pad(k, (0, 0, 0, n_pad_size))
        v = torch.nn.functional.pad(v, (0, 0, 0, n_pad_size))

        o = torch.empty_like(q)
        M = torch.empty((bs, n_heads, n_ctx + m_pad_size), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(n_ctx, BLOCK_M), bs * n_heads, 1)
        _attn_fwd[grid](
            TensorDescriptor.from_tensor(q, [1, 1, BLOCK_M, HEAD_DIM_K]),
            TensorDescriptor.from_tensor(k, [1, 1, BLOCK_N, HEAD_DIM_K]),
            TensorDescriptor.from_tensor(v, [1, 1, BLOCK_N, HEAD_DIM_K]),
            sinks,
            sm_scale,
            M,
            TensorDescriptor.from_tensor(o, [1, 1, BLOCK_M, HEAD_DIM_K]),
            start_q,
            q.shape[0],
            q.shape[1],
            k.shape[1],
            N_Q_CTX=n_ctx + m_pad_size,
            N_KV_CTX=n_kv_ctx,
            HEAD_DIM=HEAD_DIM_K,
            BANDWIDTH=bandwidth,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, sinks, o, M, start_q)
        ctx.sm_scale = sm_scale
        ctx.bandwidth = bandwidth
        ctx.m_pad_size = m_pad_size
        ctx.n_pad_size = n_pad_size
        ctx.n_ctx = n_ctx
        ctx.n_kv_ctx = n_kv_ctx

        o = o[:, :, :n_ctx, :].transpose(1, 2).contiguous() # (bs, n_ctx, n_heads, HEAD_DIM_V)
        return o
    
    @staticmethod
    def backward(ctx, do):
        q, k, v, sinks, o, M, start_q = ctx.saved_tensors
        
        bandwidth = ctx.bandwidth
        sm_scale = ctx.sm_scale
        m_pad_size = ctx.m_pad_size
        n_pad_size = ctx.n_pad_size
        n_ctx = ctx.n_ctx
        n_kv_ctx = ctx.n_kv_ctx
        
        bs, n_heads, n_ctx_padded, HEAD_DIM_Q = q.shape
        bs, n_kv_heads, n_kv_ctx_padded, HEAD_DIM_K = k.shape
        _, _, _, HEAD_DIM_V = v.shape
        
        do = do.transpose(1, 2).contiguous() # (bs, n_heads, n_ctx, HEAD_DIM_Q)
        # Pad do to match padded dimensions
        do = torch.nn.functional.pad(do, (0, 0, 0, m_pad_size))
        
        # Step 0: Initialize the gradients for dq, dk, dv, dsinks
        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dsinks = torch.zeros_like(sinks, dtype=torch.float32)  if sinks is not None else None
        
        BLOCK_M, BLOCK_N = 64, 64
        grid = (triton.cdiv(n_ctx, BLOCK_M), bs * n_heads, 1)
        
        # pre-compute D = sum(dO * O)
        D = torch.empty_like(M)
        _attn_bwd_precompute_D[grid](
            TensorDescriptor.from_tensor(D, [1, 1, BLOCK_M]),
            TensorDescriptor.from_tensor(do, [1, 1, BLOCK_M, HEAD_DIM_Q]),
            TensorDescriptor.from_tensor(o, [1, 1, BLOCK_M, HEAD_DIM_Q]),
            n_heads,
            n_ctx_padded,
            HEAD_DIM_Q,
            BLOCK_M,
        )
        
        # Backward pass
        _attn_bwd[grid](
            TensorDescriptor.from_tensor(q, [1, 1, BLOCK_M, HEAD_DIM_Q]),
            TensorDescriptor.from_tensor(k, [1, 1, BLOCK_N, HEAD_DIM_K]),
            TensorDescriptor.from_tensor(v, [1, 1, BLOCK_N, HEAD_DIM_V]),
            sinks,
            sm_scale,
            TensorDescriptor.from_tensor(do, [1, 1, BLOCK_M, HEAD_DIM_Q]),
            TensorDescriptor.from_tensor(dq, [1, 1, BLOCK_M, HEAD_DIM_Q]),
            dk,
            dv,
            dsinks,
            M,
            D,
            start_q,
            q.shape[0],
            q.shape[1],
            k.shape[1],
            N_Q_CTX=n_ctx_padded,
            N_KV_CTX=n_kv_ctx_padded,
            HEAD_DIM=HEAD_DIM_Q,
            BANDWIDTH=bandwidth,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        
        dq = dq[:, :, :n_ctx, :]
        dk = dk[:, :, :n_kv_ctx, :]
        dv = dv[:, :, :n_kv_ctx, :]
        return dq, dk.to(k.dtype), dv.to(v.dtype), dsinks.to(sinks.dtype), None, None, None


attention = _attention.apply
def triton_flash_attention(
    module,
    query,
    key,
    value,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    return attention(query, key, value, module.sinks, scaling, module.sliding_window, None), None