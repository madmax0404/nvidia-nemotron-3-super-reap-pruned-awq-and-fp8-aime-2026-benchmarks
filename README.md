# About the repo

Benchmarks of the [REAP](https://arxiv.org/html/2510.13999v1)-pruned & fine-tuned [NVIDIA Nemotron 3 Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) under two
post-training quantization schemes on MathArena AIME 2026.

- BF16: https://huggingface.co/Max-and-Omnis/Nemotron-3-Super-64B-A12B-Math-REAP-BF16
- FP8: https://huggingface.co/Max-and-Omnis/Nemotron-3-Super-64B-A12B-Math-REAP-FP8
- AWQ: https://huggingface.co/Max-and-Omnis/Nemotron-3-Super-64B-A12B-Math-REAP-AWQ

## Data

[AIMO3's public reference problems](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/data) and [AstralMath-v1](https://huggingface.co/datasets/nguyen599/AstralMath-v1) have been used.

## REAP Pruning

[The Nvidia's original model](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) has been pruned using [Cerebras's REAP](https://arxiv.org/html/2510.13999v1) method to have the model's experts reduced from 512 to 256 as well as its MTP layer removed, effectively reducing its size to half yet retaining its math & tool-integrated reasoning abilities.

## Fine-tuning

The REAP-pruned model then has been LoRA RL fine-tuned briefly with 2 problems of the AIMO3 data + AstralMath-v1 data where success_rate <= 0.1, totalling ~270 problems, for a single epoch.

## Benchmark Setup

- **Patch**: To run the REAP-pruned models on vllm, run: `uv run patches/vllm_grouped_topk.py`
- **Hardware**: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120)
- **Server**: vLLM 0.19.1 (torch 2.10.0+cu128, FlashAttention 2 backend via PTX forward-compat)
- **Benchmark**: [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026) — 30 problems, 4 attempts per problem (120 total), temperature=1.0, top_p=0.95
- **Prompt**: instruction placed in the **system** role (user-role placement gave materially worse results):
  > Put your final answer within `\boxed{}`. The answer is an integer between 0 and 999 inclusive.

| Variant | Model | Quantization | `max-model-len` |
|---------|-------|--------------|------------------|
| AWQ     | [Max-and-Omnis/Nemotron-3-Super-Math-REAP-AWQ](https://huggingface.co/Max-and-Omnis/Nemotron-3-Super-Math-REAP-AWQ) | W4A16 (AWQ INT4) | 262,144 |
| FP8     | [Max-and-Omnis/Nemotron-3-Super-Math-REAP-FP8](https://huggingface.co/Max-and-Omnis/Nemotron-3-Super-Math-REAP-FP8) | W8A8 (FP8 dynamic) | 262,144 |

Run: `bash src/start_vllm_server.sh` (or `_fp8.sh`), then `python src/benchmark_aime_2026.py` (or `_fp8.py`).

## Results

### Vram Usage

- AWQ: ~43GB
- FP8 dynamic: ~72GB

### Accuracy

| Variant | avg@4 | pass@4 | tool use |
|---------|----------------------:|---------------------:|---:|
| [Base model](https://matharena.ai/?view=problem&comp=aime--aime_2026) | **0.9000** | n\a | no |
| [AWQ](notebooks/002_results_eda.ipynb)     | **0.9083** | **0.9333** | no |
| [FP8](notebooks/003_results_eda_fp8.ipynb)     | **0.9167** | **0.9667** | no |

FP8 is ~1% better on avg@4 and +1 problem better on pass@4. The pass@4 gap is the more robust signal: FP8's better numerics cracked problem 10 (2/4 correct vs AWQ's 0/4). Problem 15 remains unsolved by both (0/4 each).

### Throughput

FP8 is **~40% slower** than AWQ in this decode-heavy workload. Reason: this is memory-bandwidth-bound decode, and W4 weights transfer half the bytes of W8 per forward step. The A8-vs-A16 saving barely matters because activations are ~10⁴× smaller than weights at low batch. FP8 tensor core compute advantage doesn't cash in when the GPU is waiting on memory. However, the FP8 model converges to answers faster, negating the slow throughput to a degree.

### Takeaway

- **AWQ** for throughput: 40% faster, quality drop is ~1 avg@4 point.
- **FP8 dynamic** for quality: +1 solvable problem, 40% throughput tax. Converges faster.
- Instruction placement matters for this model: **system-role +5% absolute over user-role prefix** on this benchmark. User-role placement leaks the instruction into the reasoning trace; system-role keeps it as a directive.

## Benchmark Result Data

- `data/aime_2026_benchmark.csv` — per-attempt AWQ results
- `data/aime_2026_benchmark_fp8.csv` — per-attempt FP8 results

Columns: `problem_idx`, `attempt_idx`, `expected_answer`, `reasoning`, `answer`, `finish_reason`, `prompt_tokens`, `completion_tokens`, `error`.

## Citations

```
@misc{balunovic_srimatharena_2025,
  title = {MathArena: Evaluating LLMs on Uncontaminated Math Competitions},
  author = {Mislav Balunović and Jasper Dekoninck and Ivo Petrov and Nikola Jovanović and Martin Vechev},
  copyright = {MIT},
  url = {https://matharena.ai/},
  publisher = {SRI Lab, ETH Zurich},
  month = feb,
  year = {2025},
}

@misc{nguyen2026astralmath,
  title={AstralMath-v1: A Large-Scale Multi-Model Tool-Integrated Reasoning Dataset for Mathematical Problem Solving},
  author={Nguyen Nguyen},
  year={2026},
  url={https://huggingface.co/datasets/nguyen599/AstralMath-v1},
}

@misc{nvidia_nemotron_3_2025,
  title  = {NVIDIA Nemotron 3: Efficient and Open Intelligence},
  author = {{NVIDIA}},
  year   = {2025},
  url    = {https://arxiv.org/abs/2512.20856},
  note   = {White Paper}
}
```


## License

Code: Apache License 2.0 — see [LICENSE](LICENSE).

This repository evaluates models on the **AIME 2026 dataset** 
([MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026)), 
which is licensed under **CC BY-NC-SA 4.0**. The dataset is not redistributed 
in this repository; it is loaded at runtime via the Hugging Face datasets API. 
Benchmark results derived from the dataset are presented for research and 
evaluation purposes.

The CSV files in `data/` contain model outputs derived from the 
AIME 2026 dataset (CC BY-NC-SA 4.0). Specifically:
- The `expected_answer` column reproduces gold answers from the dataset.
- The `reasoning` column may contain incidental verbatim quotes of 
  problem statements as part of the model's chain-of-thought.

These derivative outputs are also made available under CC BY-NC-SA 4.0 
to comply with the upstream dataset's ShareAlike requirement.

[AIMO3's public reference problems](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/data) 
(licensed under CC BY-SA 4.0) and [AstralMath-v1](https://huggingface.co/datasets/nguyen599/AstralMath-v1)(licensed under CC BY-SA 4.0) 
have been used to train the evaluated model. **Neither dataset is included or 
redistributed in this repository.** Per the AIMO3 Competition Rules §4.b.1, 
the AIMO3 Competition Data may not be redistributed; users wishing to 
reproduce training must register on Kaggle and access the data directly.

© 2026 Max & Omnis Inc. (https://www.maxandomnis.com/)