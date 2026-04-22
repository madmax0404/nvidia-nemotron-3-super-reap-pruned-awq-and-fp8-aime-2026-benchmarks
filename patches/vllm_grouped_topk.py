"""
Patch vLLM's fused grouped_topk to skip the CUDA kernel when experts_per_group > 128.

The fused grouped_topk CUDA kernel crashes with illegal memory access for large
experts_per_group values (e.g., 192 or 256 routed experts with n_group=1). This patch
adds a guard that falls back to the PyTorch implementation.

Usage:
    python patches/vllm_grouped_topk.py          # apply patch
    python patches/vllm_grouped_topk.py --revert  # revert patch
"""

import sys
from pathlib import Path

TARGET = Path(
    ".venv/lib/python3.12/site-packages/vllm/"
    "model_executor/layers/fused_moe/router/grouped_topk_router.py"
)

ORIGINAL = """\
    if (
        envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK
        and current_platform.is_cuda()
        and num_expert_group <= 32
        and topk <= 32
        and e_score_correction_bias is not None
    ):"""

PATCHED = """\
    num_experts = gating_output.size(-1)
    experts_per_group = num_experts // max(num_expert_group, 1)
    if (
        envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK
        and current_platform.is_cuda()
        and num_expert_group <= 32
        and topk <= 32
        and experts_per_group <= 128
        and e_score_correction_bias is not None
    ):"""


def main():
    revert = "--revert" in sys.argv
    path = Path(__file__).resolve().parent.parent / TARGET

    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    content = path.read_text()

    if revert:
        if ORIGINAL in content:
            print("Already reverted.")
            return
        if PATCHED not in content:
            print("ERROR: cannot find patched code to revert.")
            sys.exit(1)
        content = content.replace(PATCHED, ORIGINAL)
        path.write_text(content)
        print("Reverted.")
    else:
        if PATCHED in content:
            print("Already patched.")
            return
        if ORIGINAL not in content:
            print("ERROR: cannot find original code to patch.")
            sys.exit(1)
        content = content.replace(ORIGINAL, PATCHED)
        path.write_text(content)
        print("Patched.")


if __name__ == "__main__":
    main()
