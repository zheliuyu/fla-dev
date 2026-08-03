# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""CP chunk_delta_h pre-process kernels adapted for triton-ascend on Ascend NPU."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from fla.ops.cp.comm import all_gather_into_tensor
from fla.ops.utils.op import exp2
from fla.utils import input_guard
from fla.utils.ascend_ub_manager import (
    ASCEND_MAX_GRID_DIM,
    compute_row_tile_block_size,
    max_grid_axis_chunks,
)

if TYPE_CHECKING:
    from fla.ops.cp.context import FLACPContext

# Peak live fp32: b_h[64,BV] + b_w[64,64] + b_v[64,BV] + b_k[64,64] during recurrence.
_PREPROCESS_MEM_MULT = 8.0
_SAFETY_MARGIN = 0.80
_FALLBACK_BLOCK = 16
# Larger tiles amortize the per-iteration scalar/address-gen cost of the
# sequential chunk recurrence (these kernels are scalar-bound on NPU).
# 64 is the UB-safe ceiling for D128 (mem_mult model); D64 caps at pow2(V)=64.
_MAX_BLOCK = 64


def _get_block_size(K: int, V: int) -> int:
    """Compute UB-safe block size for pre-process kernels."""
    return compute_row_tile_block_size(
        min(K, 64),
        V,
        _PREPROCESS_MEM_MULT,
        tiling_row=False,
        safety_margin=_SAFETY_MARGIN,
        fallback=_FALLBACK_BLOCK,
        min_block=16,
        max_block=min(_MAX_BLOCK, triton.next_power_of_2(V)),
    )


def _launch_preprocess_kernel(kernel, *, nv_chunks: int, nh_total: int, kernel_kwargs: dict) -> None:
    """Launch pre-process kernel with grid-axis chunking for Ascend."""
    max_nv = max_grid_axis_chunks(nv_chunks, nh_total, max_grid=ASCEND_MAX_GRID_DIM)
    for v_off in range(0, nv_chunks, max_nv):
        v_len = min(max_nv, nv_chunks - v_off)
        kernel_kwargs['V_OFFSET'] = v_off
        max_nh = max_grid_axis_chunks(nh_total, v_len, max_grid=ASCEND_MAX_GRID_DIM)
        for nh_off in range(0, nh_total, max_nh):
            nh_len = min(max_nh, nh_total - nh_off)
            kernel_kwargs['NH_OFFSET'] = nh_off
            kernel[(v_len, nh_len)](**kernel_kwargs)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_BG': lambda args: args['bg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def pre_process_fwd_kernel_merged_npu(
    k,
    v,
    w,
    g,
    gk,
    bg,
    u,
    hm,
    cu_seqlens,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    MULTI_SEQS: tl.constexpr,
    V_OFFSET: tl.constexpr = 0,
    NH_OFFSET: tl.constexpr = 0,
):
    """Forward pre-process kernel for CP on NPU.

    Computes h (K x V) and m (K x K) matrices for cross-rank state passing.
    Grid: (V/BLOCK_SIZE + K/BLOCK_SIZE, HV) with host-side chunking via V_OFFSET/NH_OFFSET.
    """
    i_col = tl.program_id(0) + V_OFFSET
    i_h = tl.program_id(1) + NH_OFFSET

    if MULTI_SEQS:
        i_n = tl.program_id(2)
        hm += i_n * HV * K * (K + V) + i_h * K * (K + V)
    else:
        i_n = 0
        hm += i_h * K * (K + V)

    # Save the packed length before varlen overwrites T (used for g strides).
    T_seq = T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    # Determine if this block handles h (V part) or m (K part)
    is_h_part = i_col * BLOCK_SIZE < V

    # For DPLR (USE_BG), w and bg share the same head dim H as k/ag.
    # For GDN/KDA, w has head dim HV (same as v).
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
        bg += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        # g is host-transposed to [B, HV, T] (G_T_CONTIG): stride-1 loads along T.
        if IS_VARLEN:
            g += (bos + i_h.to(tl.int64) * T_seq).to(tl.int64)
        else:
            g += ((i_n * HV + i_h).to(tl.int64) * T_seq)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    stride_k = H * K
    stride_w = H * K if USE_BG else HV * K

    if is_h_part:
        # ====== Stage 1: Compute h (K x V) ======
        v += ((bos * HV + i_h) * V).to(tl.int64)
        if USE_BG:
            u += ((bos * HV + i_h) * V).to(tl.int64)
        stride_v = HV * V
        i_v = i_col

        # Initialize h accumulators
        b_h1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        o_k1 = tl.arange(0, 64)
        o_k2 = 64 + o_k1
        o_k3 = 128 + o_k1
        o_k4 = 192 + o_k1

        # Main recurrence for h
        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            # Compute decayed v
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v_decay = tl.dot(b_w, b_h1.to(b_w.dtype))
            if K > 64:
                p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v_decay += tl.dot(b_w, b_h2.to(b_w.dtype))
            if K > 128:
                p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v_decay += tl.dot(b_w, b_h3.to(b_w.dtype))
            if K > 192:
                p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v_decay += tl.dot(b_w, b_h4.to(b_w.dtype))

            p_v = tl.make_block_ptr(v, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
            if USE_BG:
                b_v_orig = tl.load(p_v, boundary_check=(0, 1))
                p_u = tl.make_block_ptr(u, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
                b_v = b_v_decay + tl.load(p_u, boundary_check=(0, 1))
            else:
                b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v_decay

            last_idx = min((i_t + 1) * BT, T) - 1

            # Apply g decay
            if USE_G:
                b_g_last = tl.load(g + last_idx).to(tl.float32)
                p_g = tl.make_block_ptr(g, (T,), (1,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_h1 *= b_g_last
                if K > 64:
                    b_h2 *= b_g_last
                if K > 128:
                    b_h3 *= b_g_last
                if K > 192:
                    b_h4 *= b_g_last

            # Apply gk decay
            if USE_GK:
                p_gk_last = gk + last_idx * HV * K
                b_gk_last1 = tl.load(p_gk_last + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
                b_h1 *= exp2(b_gk_last1)[:, None]
                if K > 64:
                    b_gk_last2 = tl.load(p_gk_last + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                    b_h2 *= exp2(b_gk_last2)[:, None]
                if K > 128:
                    b_gk_last3 = tl.load(p_gk_last + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                    b_h3 *= exp2(b_gk_last3)[:, None]
                if K > 192:
                    b_gk_last4 = tl.load(p_gk_last + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                    b_h4 *= exp2(b_gk_last4)[:, None]
            b_v = b_v.to(k.dtype.element_ty)

            # Update h
            p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_BG:
                p_bg = tl.make_block_ptr(bg, (K, T), (1, stride_k), (0, i_t * BT), (64, BT), (0, 1))
                b_bg = tl.load(p_bg, boundary_check=(0, 1))
                b_h1 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 64:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    p_bg = tl.make_block_ptr(bg, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                    b_bg = tl.load(p_bg, boundary_check=(0, 1))
                    b_h2 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 128:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    p_bg = tl.make_block_ptr(bg, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                    b_bg = tl.load(p_bg, boundary_check=(0, 1))
                    b_h3 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
                if K > 192:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    p_bg = tl.make_block_ptr(bg, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                    b_bg = tl.load(p_bg, boundary_check=(0, 1))
                    b_h4 += tl.dot(b_k, b_v_orig.to(b_k.dtype)) + tl.dot(b_bg, b_v)
            else:
                b_h1 += tl.dot(b_k, b_v)
                if K > 64:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (64, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_h2 += tl.dot(b_k, b_v)
                if K > 128:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (128, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_h3 += tl.dot(b_k, b_v)
                if K > 192:
                    p_k = tl.make_block_ptr(k, (K, T), (1, stride_k), (192, i_t * BT), (64, BT), (0, 1))
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_h4 += tl.dot(b_k, b_v)

        # Store h results
        stride_hm_kv = K + V
        p_h1 = tl.make_block_ptr(hm, (K, stride_hm_kv), (stride_hm_kv, 1), (0, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(hm, (K, stride_hm_kv), (stride_hm_kv, 1),
                                     (64, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(hm, (K, stride_hm_kv), (stride_hm_kv, 1),
                                     (128, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(hm, (K, stride_hm_kv), (stride_hm_kv, 1),
                                     (192, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))
    else:
        # ====== Stage 2: Compute m (K x K) ======
        # i_col is for m part, map to K dimension
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        # Following stage2 kernel design:
        # - BK1 is the full K dimension (next_power_of_2(K))
        # - BLOCK_SIZE is the column block size (like BK2=32 in stage2)
        # Each block computes a (BK1, BLOCK_SIZE) sub-matrix of m
        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        # Initialize b_m as zeros and use += accumulation pattern for NPU compatibility
        b_m = tl.zeros([BK1, BLOCK_SIZE], dtype=tl.float32)
        # Add identity matrix contribution
        b_m += tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for i_t in range(NT):
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T
            # Load k and w with full BK1 rows
            if USE_BG:
                p_k = tl.make_block_ptr(bg, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            else:
                p_k = tl.make_block_ptr(k, (T, K), (stride_k, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx).to(tl.float32)
                p_g = tl.make_block_ptr(g, (T,), (1,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_gk_last = tl.load(gk + last_idx * HV * K + row, mask=(row < K), other=0.).to(tl.float32)
                b_gk_last = exp2(b_gk_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

            # Compute m update using += pattern for NPU compatibility
            if USE_BG:
                b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
                b_m_i = b_diag + b_kw
            else:
                b_kw = tl.dot(tl.trans(b_k.to(b_w.dtype)), b_w)
                b_m_i = b_diag - b_kw
            # Use += to accumulate, maintaining address space consistency
            b_m += tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32)) - b_m

        # Store m result
        stride_hm_kv = K + V
        p_m = tl.make_block_ptr(hm, (K, stride_hm_kv), (stride_hm_kv, 1),
                                (0, V + i_k_col * BLOCK_SIZE), (BK1, BLOCK_SIZE), (1, 0))
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'HAS_H0': lambda args: args['h0'] is not None,
})
@triton.jit(do_not_specialize=['pre_or_post_num_ranks', 'rank', 'NUM_SEQ_ENTRIES'])
def merge_fwd_bwd_kernel_npu(
    h,
    ag_hm,
    pre_or_post_num_ranks,
    rank,
    seq_offsets,
    init_offsets,
    h0_seq_ids,
    h0,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
    FORWARD: tl.constexpr,
    INTRACARD_MODE: tl.constexpr,
    NUM_SEQ_ENTRIES,
    HAS_H0: tl.constexpr,
    STATE_V_FIRST: tl.constexpr = False,
):
    """Merge kernel for CP on NPU.

    Merges h/m matrices across ranks or sub-sequences.
    Grid: (V/BV, HV) for CP mode, (V/BV, NUM_SEQ_ENTRIES, HV) for intracard mode.
    """
    i_v = tl.program_id(0)
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if INTRACARD_MODE:
        i_seq = tl.program_id(1)
        i_h = tl.program_id(2)

        if i_seq >= NUM_SEQ_ENTRIES:
            return

        ss_start = tl.load(seq_offsets + i_seq).to(tl.int32)
        ss_end = tl.load(seq_offsets + i_seq + 1).to(tl.int32)
        init_base = tl.load(init_offsets + i_seq).to(tl.int32)
        num_subseqs = ss_end - ss_start

        stride_hm_s = HV * K * (V + K)
        stride_hm_h = K * (V + K)

        if HAS_H0:
            orig_seq_id = tl.load(h0_seq_ids + i_seq).to(tl.int32)
            if STATE_V_FIRST:
                p_h0 = h0 + (orig_seq_id * HV + i_h) * V * K + o_v[:, None] * K + o_k[None, :]
                b_h = tl.load(p_h0, mask=m_v[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            else:
                p_h0 = h0 + (orig_seq_id * HV + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
                b_h = tl.load(p_h0, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        else:
            if STATE_V_FIRST:
                b_h = tl.zeros([BV, BK], dtype=tl.float32)
            else:
                b_h = tl.zeros([BK, BV], dtype=tl.float32)

        for idx in range(num_subseqs):
            i_ss = ss_start + idx
            base = i_ss * stride_hm_s + i_h * stride_hm_h

            p_he = ag_hm + base + o_k[:, None] * (V + K) + o_v[None, :]
            b_he = tl.load(p_he, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
            p_m = ag_hm + base + V + o_k[:, None] * (V + K) + o_k[None, :]
            b_m = tl.load(p_m, mask=m_k[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            if STATE_V_FIRST:
                # Ensure b_h stays in consistent address space by creating a new accumulator
                b_h_new = tl.dot(b_h.to(tl.float32), tl.trans(b_m)) + tl.trans(b_he)
                b_h = b_h_new
            else:
                b_h_new = tl.dot(b_m.to(tl.float32), b_h.to(tl.float32)) + b_he.to(tl.float32)
                b_h = b_h_new

            if idx < num_subseqs - 1:
                init_idx = init_base + idx
                stride_init = HV * K * V
                if STATE_V_FIRST:
                    p_out = h + init_idx * stride_init + i_h * V * K + o_v[:, None] * K + o_k[None, :]
                    m_out = m_v[:, None] & m_k[None, :]
                else:
                    p_out = h + init_idx * stride_init + i_h * K * V + o_k[:, None] * V + o_v[None, :]
                    m_out = m_k[:, None] & m_v[None, :]
                tl.store(p_out, b_h.to(p_out.dtype.element_ty), mask=m_out)
    else:
        # CP mode
        i_h = tl.program_id(1)
        num_ranks = pre_or_post_num_ranks.to(tl.int32)
        h += i_h * K * V
        ag_hm += i_h * K * (K + V)
        stride = HV * K * (K + V)
        if STATE_V_FIRST:
            b_h = tl.zeros([BV, BK], dtype=tl.float32)
        else:
            b_h = tl.zeros([BK, BV], dtype=tl.float32)
        for idx in range(num_ranks):
            if FORWARD:
                cur_rank = rank - num_ranks + idx
            else:
                cur_rank = rank + num_ranks - idx
            p_ag_h = ag_hm + cur_rank * stride + o_k[:, None] * (K + V) + o_v[None, :]
            b_ag_h = tl.load(p_ag_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
            p_ag_m = ag_hm + cur_rank * stride + V + o_k[:, None] * (K + V) + o_k[None, :]
            b_ag_m = tl.load(p_ag_m, mask=m_k[:, None] & m_k[None, :], other=0.0)
            if STATE_V_FIRST:
                # Create new accumulator to maintain consistent address space
                b_h_new = tl.dot(b_h.to(tl.float32), tl.trans(b_ag_m).to(tl.float32)) + tl.trans(b_ag_h).to(tl.float32)
                b_h = b_h_new
            else:
                b_h_new = tl.dot(b_ag_m.to(tl.float32), b_h.to(tl.float32)) + b_ag_h.to(tl.float32)
                b_h = b_h_new
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = m_v[:, None] & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & m_v[None, :]
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=m_h)


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def pre_process_bwd_kernel_merged_npu(
    q,
    k,
    w,
    g,
    gk,
    do,
    dhm,
    dv,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BK1: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_BG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    V_OFFSET: tl.constexpr = 0,
    NH_OFFSET: tl.constexpr = 0,
):
    """Backward pre-process kernel for CP on NPU.

    Computes dh (K x V) and dm (K x K) matrices for cross-rank gradient passing.
    Grid: (V/BLOCK_SIZE + K/BLOCK_SIZE, HV) with host-side chunking via V_OFFSET/NH_OFFSET.
    """
    i_col = tl.program_id(0) + V_OFFSET
    i_h = tl.program_id(1) + NH_OFFSET
    i_n = 0

    # Save the packed length before varlen overwrites T (used for g strides).
    T_seq = T
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)
        NT = tl.cdiv(T, BT)

    is_dh_part = i_col * BLOCK_SIZE < V

    q += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    k += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    if USE_BG:
        w += ((bos * H + i_h // (HV // H)) * K).to(tl.int64)
    else:
        w += ((bos * HV + i_h) * K).to(tl.int64)
    if USE_G:
        # g is host-transposed to [B, HV, T] (G_T_CONTIG): stride-1 loads along T.
        # bwd always has i_n == 0, so bos already encodes the batch offset.
        g += (bos + i_h.to(tl.int64) * T_seq).to(tl.int64)
    if USE_GK:
        gk += ((bos * HV + i_h) * K).to(tl.int64)
    dhm += i_h * K * (V + K)
    stride_qk = H * K
    stride_w = H * K if USE_BG else HV * K

    if is_dh_part:
        # ====== Stage 1: Compute dh (K x V) ======
        do += ((bos * HV + i_h) * V).to(tl.int64)
        dv += ((bos * HV + i_h) * V).to(tl.int64)
        stride_v = HV * V
        i_v = i_col

        b_dh1 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 64:
            b_dh2 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 128:
            b_dh3 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)
        if K > 192:
            b_dh4 = tl.zeros([64, BLOCK_SIZE], dtype=tl.float32)

        o_k1 = tl.arange(0, 64)
        o_k2 = 64 + o_k1
        o_k3 = 128 + o_k1
        o_k4 = 192 + o_k1

        for i_t in range(NT - 1, -1, -1):
            last_idx = min((i_t + 1) * BT, T) - 1
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T

            if USE_G:
                bg_last = tl.load(g + last_idx).to(tl.float32)
                p_g = tl.make_block_ptr(g, (T,), (1,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)

            p_dv = tl.make_block_ptr(dv, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (stride_v, 1), (i_t * BT, i_v * BLOCK_SIZE), (BT, BLOCK_SIZE), (1, 0))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # Fold gate*exp and scale into do once so K-slabs skip per-slab q gating.
            # dot(q * g_exp, do) * scale == dot(q, do * g_exp * scale).
            if USE_G:
                if USE_BG:
                    b_do = b_do * b_g_exp[:, None]
                else:
                    b_do = b_do * (b_g_exp * scale)[:, None]
            elif not USE_BG:
                b_do = b_do * scale

            p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                p_gk_last = gk + last_idx * HV * K
                b_gk_last1 = tl.load(p_gk_last + o_k1, mask=(o_k1 < K), other=0.).to(tl.float32)
            b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

            if K > 64:
                p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 64), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    b_gk_last2 = tl.load(p_gk_last + o_k2, mask=(o_k2 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

            if K > 128:
                p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 128), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    b_gk_last3 = tl.load(p_gk_last + o_k3, mask=(o_k3 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

            if K > 192:
                p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 192), (BT, 64), (1, 0))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                if USE_GK:
                    b_gk_last4 = tl.load(p_gk_last + o_k4, mask=(o_k4 < K), other=0.).to(tl.float32)
                b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

            if USE_G:
                b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
            b_dv += tl.load(p_dv, boundary_check=(0, 1))

            p_w = tl.make_block_ptr(w, (K, T), (1, stride_w), (0, i_t * BT), (64, BT), (0, 1))
            p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (0, i_t * BT), (64, BT), (0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            if USE_G:
                b_dh1 *= bg_last_exp
            if USE_GK:
                b_dh1 *= exp2(b_gk_last1[:, None])
            if USE_BG:
                b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
            else:
                b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 64:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (64, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_w), (64, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh2 *= bg_last_exp
                if USE_GK:
                    b_dh2 *= exp2(b_gk_last2[:, None])
                if USE_BG:
                    b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 128:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (128, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_w), (128, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh3 *= bg_last_exp
                if USE_GK:
                    b_dh3 *= exp2(b_gk_last3[:, None])
                if USE_BG:
                    b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

            if K > 192:
                p_q = tl.make_block_ptr(q, (K, T), (1, stride_qk), (192, i_t * BT), (64, BT), (0, 1))
                p_w = tl.make_block_ptr(w, (K, T), (1, stride_w), (192, i_t * BT), (64, BT), (0, 1))
                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_w = tl.load(p_w, boundary_check=(0, 1))
                if USE_G:
                    b_dh4 *= bg_last_exp
                if USE_GK:
                    b_dh4 *= exp2(b_gk_last4[:, None])
                if USE_BG:
                    b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) + tl.dot(b_w, b_dv.to(b_w.dtype))
                else:
                    b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) - tl.dot(b_w, b_dv.to(b_w.dtype))

        # Store dh results
        p_dh1 = tl.make_block_ptr(dhm, (K, V + K), (V + K, 1), (0, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(dhm, (K, V + K), (V + K, 1), (64, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dhm, (K, V + K), (V + K, 1), (128, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dhm, (K, V + K), (V + K, 1), (192, i_v * BLOCK_SIZE), (64, BLOCK_SIZE), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))
    else:
        # ====== Stage 2: Compute dm (K x K) ======
        i_k_col = i_col - tl.cdiv(V, BLOCK_SIZE)

        row = tl.arange(0, BK1)
        col = tl.arange(0, BLOCK_SIZE) + i_k_col * BLOCK_SIZE

        # Initialize b_m as zeros and use += accumulation pattern for NPU compatibility
        b_m = tl.zeros([BK1, BLOCK_SIZE], dtype=tl.float32)
        b_m += tl.where(row[:, None] == col[None, :], 1.0, 0.0)

        for _i_t in range(NT):
            i_t = NT - 1 - _i_t
            o_t = i_t * BT + tl.arange(0, BT)
            m_t = o_t < T

            p_k = tl.make_block_ptr(k, (T, K), (stride_qk, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            p_w = tl.make_block_ptr(w, (T, K), (stride_w, 1), (i_t * BT, 0), (BT, BK1), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))

            last_idx = min((i_t + 1) * BT, T) - 1

            if USE_G:
                b_g_last = tl.load(g + last_idx).to(tl.float32)
                p_g = tl.make_block_ptr(g, (T,), (1,), (i_t * BT,), (BT,), (0,))
                b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
                b_k = b_k * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_g_last, 0.0)
            elif USE_GK:
                b_gk_last = tl.load(gk + last_idx * HV * K + row, mask=(row < K), other=0.).to(tl.float32)
                b_gk_last = exp2(b_gk_last)
                b_diag = tl.where(row[:, None] == row[None, :], b_gk_last[:, None], 0.0)
            else:
                b_diag = tl.where(row[:, None] == row[None, :], 1.0, 0.0)

            b_kw = tl.dot(tl.trans(b_w), b_k.to(b_w.dtype))
            if USE_BG:
                b_m_i = b_diag + b_kw
            else:
                b_m_i = b_diag - b_kw
            # Use += to accumulate, maintaining address space consistency
            b_m += tl.dot(b_m_i.to(tl.float32), b_m.to(tl.float32)) - b_m

        p_m = tl.make_block_ptr(dhm, (K, V + K), (V + K, 1), (0, V + i_k_col * BLOCK_SIZE), (BK1, BLOCK_SIZE), (1, 0))
        tl.store(p_m, b_m.to(p_m.dtype.element_ty), boundary_check=(0, 1))


@input_guard
def chunk_gated_delta_rule_fwd_h_pre_process_npu(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    chunk_size: int = 64,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pre-process for CP on NPU."""
    if context is None or context.group is None:
        return initial_state
    assert initial_state is None, "When enable CP, the provided initial_state must be None."
    rank = dist.get_rank(group=context.group)

    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size
    BK = triton.next_power_of_2(K)

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    hm = k.new_zeros(HV, K, (V + K), dtype=torch.float32)
    if state_v_first:
        initial_state = k.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        initial_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)

    if g is not None and HV > 1:
        # G_T_CONTIG: transpose g to [B, HV, T] so kernels load stride-1 along T.
        g = g.transpose(1, 2).contiguous()

    if not context.is_last_rank:
        BLOCK_SIZE = _get_block_size(K, V)
        nv_chunks = triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE)
        nh_total = HV
        _launch_preprocess_kernel(
            pre_process_fwd_kernel_merged_npu,
            nv_chunks=nv_chunks,
            nh_total=nh_total,
            kernel_kwargs={
                'k': k,
                'v': u if v is None else v,
                'w': w,
                'g': g,
                'gk': gk,
                'bg': bg,
                'u': u,
                'hm': hm,
                'cu_seqlens': cu_seqlens[-2:],
                'T': T,
                'H': H,
                'HV': HV,
                'K': K,
                'V': V,
                'BT': BT,
                'BK1': BK,
                'BLOCK_SIZE': BLOCK_SIZE,
                'MULTI_SEQS': False,
                'V_OFFSET': 0,
                'NH_OFFSET': 0,
            },
        )

    ag_hm, _ = all_gather_into_tensor(hm, group=context.group)

    if not context.is_first_rank:
        def grid(meta):
            return (triton.cdiv(V, meta['BV']), HV)
        merge_fwd_bwd_kernel_npu[grid](
            h=initial_state[0],
            ag_hm=ag_hm,
            pre_or_post_num_ranks=context.pre_num_ranks,
            rank=rank,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=_get_block_size(K, V),
            FORWARD=True,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            STATE_V_FIRST=state_v_first,
        )
    return initial_state


@input_guard
def chunk_gated_delta_rule_bwd_dhu_pre_process_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    context: FLACPContext | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pre-process for CP on NPU."""
    if context is None or context.group is None:
        return dht, initial_state
    assert dht is None, "When enable CP, the provided dht must be None."
    rank = dist.get_rank(context.group)

    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    assert K <= 256, "current kernel does not support head dimension being larger than 256."
    BK = triton.next_power_of_2(K)

    if cu_seqlens is None:
        N = B
    else:
        N = len(cu_seqlens) - 1

    dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)
    if state_v_first:
        dht = q.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        dht = q.new_zeros(N, HV, K, V, dtype=torch.float32)

    if g is not None and HV > 1:
        # G_T_CONTIG: transpose g to [B, HV, T] so kernels load stride-1 along T.
        g = g.transpose(1, 2).contiguous()

    if not context.is_first_rank:
        BLOCK_SIZE = _get_block_size(K, V)
        nv_chunks = triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE)
        nh_total = HV
        _launch_preprocess_kernel(
            pre_process_bwd_kernel_merged_npu,
            nv_chunks=nv_chunks,
            nh_total=nh_total,
            kernel_kwargs={
                'q': q,
                'k': k if bg is None else bg,
                'w': w,
                'g': g,
                'gk': gk,
                'do': do,
                'dhm': dhm,
                'dv': dv,
                'cu_seqlens': cu_seqlens[:2],
                'scale': scale,
                'T': T,
                'H': H,
                'HV': HV,
                'K': K,
                'V': V,
                'BT': BT,
                'BK1': BK,
                'BLOCK_SIZE': BLOCK_SIZE,
                'USE_BG': bg is not None,
                'V_OFFSET': 0,
                'NH_OFFSET': 0,
            },
        )

    ag_dhm, _ = all_gather_into_tensor(dhm, group=context.group)

    if not context.is_last_rank:
        def grid(meta):
            return (triton.cdiv(V, meta['BV']), HV)
        merge_fwd_bwd_kernel_npu[grid](
            h=dht[-1],
            ag_hm=ag_dhm,
            pre_or_post_num_ranks=context.post_num_ranks,
            rank=rank,
            seq_offsets=None,
            init_offsets=None,
            h0_seq_ids=None,
            h0=None,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=_get_block_size(K, V),
            FORWARD=False,
            INTRACARD_MODE=False,
            NUM_SEQ_ENTRIES=0,
            STATE_V_FIRST=state_v_first,
        )

    return dht, None
