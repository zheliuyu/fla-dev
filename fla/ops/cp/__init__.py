# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Context Parallel operators and utilities

from .comm import (
    all_gather_into_tensor,
    all_reduce_sum,
    conv_cp_send_recv_bwd,
    conv_cp_send_recv_fwd,
    send_recv_bwd,
    send_recv_fwd,
)
from .context import (
    FLACPContext,
    build_cp_context,
)

__all__ = [
    "FLACPContext",
    "all_gather_into_tensor",
    "all_reduce_sum",
    "build_cp_context",
    "conv_cp_send_recv_bwd",
    "conv_cp_send_recv_fwd",
    "send_recv_bwd",
    "send_recv_fwd",
]


# Override chunk_delta_h functions with NPU implementations if available
def _override_chunk_delta_h_for_npu():
    """Override chunk_delta_h functions with NPU implementations when running on NPU."""
    from fla.utils import IS_NPU
    if not IS_NPU:
        return

    import fla.ops.cp.chunk_delta_h as cp_chunk_delta_h
    from fla.ops.cp.backends.triton_ascend.chunk_delta_h import (
        chunk_gated_delta_rule_bwd_dhu_pre_process_npu,
        chunk_gated_delta_rule_fwd_h_pre_process_npu,
        merge_fwd_bwd_kernel_npu,
        pre_process_bwd_kernel_merged_npu,
        pre_process_fwd_kernel_merged_npu,
    )

    # Replace functions with NPU implementations
    cp_chunk_delta_h.chunk_gated_delta_rule_fwd_h_pre_process = chunk_gated_delta_rule_fwd_h_pre_process_npu
    cp_chunk_delta_h.chunk_gated_delta_rule_bwd_dhu_pre_process = chunk_gated_delta_rule_bwd_dhu_pre_process_npu
    # Replace raw kernels as well so that direct importers (e.g. white-box
    # regression tests) transparently get Ascend-compilable implementations.
    # The NPU kernels are signature-compatible (grid offsets default to 0).
    cp_chunk_delta_h.pre_process_fwd_kernel_merged = pre_process_fwd_kernel_merged_npu
    cp_chunk_delta_h.pre_process_bwd_kernel_merged = pre_process_bwd_kernel_merged_npu
    cp_chunk_delta_h.merge_fwd_bwd_kernel = merge_fwd_bwd_kernel_npu


# Apply NPU overrides
_override_chunk_delta_h_for_npu()
