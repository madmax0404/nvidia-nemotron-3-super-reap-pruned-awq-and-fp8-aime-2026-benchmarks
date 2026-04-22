"""Benchmark the vLLM-served model on MathArena/aime_2026.

Starts 30 problems * 4 attempts = 120 requests in parallel against the
server launched by ``src/start_vllm_server.sh`` and writes every attempt
(reasoning + answer) to a CSV under ``data/``.

Usage:
    python src/benchmark_aime_2026.py
"""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "nemotron-3-super"
NUM_ATTEMPTS = 4
TEMPERATURE = 1.0
TOP_P = 0.95
REQUEST_TIMEOUT = 7200
INSTRUCTION = """Put your final answer within \\boxed{}.
The answer is an integer between 0 and 999 inclusive."""

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT_DIR / "data" / "aime_2026_benchmark.csv"

FIELDNAMES = [
    "problem_idx",
    "attempt_idx",
    "expected_answer",
    "reasoning",
    "answer",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "error",
]


async def run_attempt(
    client: AsyncOpenAI,
    problem_idx: int,
    attempt_idx: int,
    problem: str,
    expected_answer: int,
) -> dict:
    base = {
        "problem_idx": problem_idx,
        "attempt_idx": attempt_idx,
        "expected_answer": expected_answer,
    }
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": problem}
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
            timeout=REQUEST_TIMEOUT,
            parallel_tool_calls=False,
        )
        message = response.choices[0].message
        usage = response.usage
        return {
            **base,
            "reasoning": message.reasoning,
            "answer": message.content,
            "finish_reason": response.choices[0].finish_reason,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "error": None,
        }
    except Exception as exc:
        return {
            **base,
            "reasoning": None,
            "answer": None,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "error": repr(exc),
        }


async def main() -> None:
    ds = load_dataset("MathArena/aime_2026")["train"]
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        run_attempt(
            client,
            problem_idx=row["problem_idx"],
            attempt_idx=attempt_idx,
            problem=row["problem"],
            expected_answer=row["answer"],
        )
        for row in ds
        for attempt_idx in range(NUM_ATTEMPTS)
    ]

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        f.flush()

        for coro in tqdm_asyncio.as_completed(tasks, desc="aime_2026"):
            result = await coro
            writer.writerow(result)
            f.flush()

    df = pd.read_csv(OUTPUT_PATH).sort_values(
        by=["problem_idx", "attempt_idx"]
    ).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)

    n_errors = int(df["error"].notna().sum())
    print(f"Saved {len(df)} attempts to {OUTPUT_PATH} ({n_errors} errors)")


if __name__ == "__main__":
    asyncio.run(main())
