"""
Rate metaphor words on [0, 1] semantic dimensions using DeepInfra.

Each line in pairs.txt defines one dimension:
  left anchor - right anchor

The output score is continuous:
  0.0 = strongly left anchor
  0.5 = neutral / ambiguous / not applicable
  1.0 = strongly right anchor

The script rates each unique metaphor word once, checkpoints to JSONL, and writes:
  - output/metaphor_word_ratings_deepinfra.jsonl
  - output/metaphor_word_ratings_deepinfra.csv
  - output/metaphor_results_v3_deepinfra_rated.json

Usage:
  C:\\Users\\aruba\\Anaconda3\\envs\\agent-code\\python.exe code\\rate_metaphors_deepinfra.py --dry-run
  $env:DEEPINFRA_API_KEY = "..."
  C:\\Users\\aruba\\Anaconda3\\envs\\agent-code\\python.exe code\\rate_metaphors_deepinfra.py --workers 4

You can also put the key in deepinfra_key.txt at the repository root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm


DEFAULT_INPUT = Path("output/metaphor_results_v3.json")
DEFAULT_PAIRS = Path("pairs.txt")
DEFAULT_API_KEY_FILE = Path("deepinfra_key.txt")
DEFAULT_OUT_JSONL = Path("output/metaphor_word_ratings_deepinfra.jsonl")
DEFAULT_OUT_CSV = Path("output/metaphor_word_ratings_deepinfra.csv")
DEFAULT_OUT_ENRICHED = Path("output/metaphor_results_v3_deepinfra_rated.json")

DEEPINFRA_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

SYSTEM_PROMPT = """You are an expert semantic annotator for a social-science study of metaphors.
You rate metaphor words that may describe the U.S. federal government in historical newspaper analysis.

For each dimension, assign a continuous score from 0 to 1:
0.0 means the metaphor word strongly implies the left anchor.
0.5 means neutral, balanced, ambiguous, or not applicable.
1.0 means the metaphor word strongly implies the right anchor.

Rate the general semantic implication of the metaphor word itself, not a specific article context.
If the metaphor word is unclear, malformed, or not a meaningful metaphor, use score 0.5 with low confidence.

Return only valid JSON. Do not wrap the JSON in markdown."""


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value or "dimension"


def load_pairs(path: Path) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            if " - " in line:
                left, right = line.split(" - ", 1)
            elif "-" in line:
                left, right = line.split("-", 1)
            else:
                raise ValueError(f"Line {line_no} in {path} is not a pair: {line!r}")
            left = left.strip()
            right = right.strip()
            if not left or not right:
                raise ValueError(f"Line {line_no} in {path} has an empty anchor: {line!r}")
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "label": f"{left} - {right}",
                    "key": slug(f"{left}_{right}"),
                }
            )
    if not pairs:
        raise ValueError(f"No dimensions found in {path}")
    return pairs


def load_metaphors(path: Path, skip_na: bool) -> tuple[list[dict[str, Any]], Counter[str]]:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list")

    counts: Counter[str] = Counter()
    for row in rows:
        metaphor = str(row.get("metaphor", "")).strip()
        if not metaphor:
            continue
        if skip_na and metaphor.upper() in {"NA", "N/A", "NONE", "NULL"}:
            continue
        counts[metaphor] += 1
    return rows, counts


def load_api_key(path: Path) -> str:
    key = os.getenv("DEEPINFRA_API_KEY", "").strip()
    if key:
        return key
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    raise RuntimeError(f"Set DEEPINFRA_API_KEY or place a key in {path}")


def build_user_prompt(metaphor: str, pairs: list[dict[str, str]]) -> str:
    dimensions = [
        {
            "dimension": pair["label"],
            "left_anchor": pair["left"],
            "right_anchor": pair["right"],
        }
        for pair in pairs
    ]
    return json.dumps(
        {
            "task": "Rate one metaphor word on every semantic dimension.",
            "metaphor": metaphor,
            "dimensions": dimensions,
            "required_json_schema": {
                "metaphor": metaphor,
                "ratings": [
                    {
                        "dimension": "exact dimension label",
                        "left_anchor": "left anchor",
                        "right_anchor": "right anchor",
                        "score": "number from 0.0 to 1.0",
                        "confidence": "number from 0.0 to 1.0",
                        "reasoning": "brief phrase, not a long explanation",
                    }
                ],
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def normalize_result(raw: dict[str, Any], metaphor: str, pairs: list[dict[str, str]]) -> dict[str, Any]:
    ratings_by_dimension = {
        str(r.get("dimension", "")).strip(): r for r in raw.get("ratings", []) if isinstance(r, dict)
    }
    normalized: list[dict[str, Any]] = []

    for pair in pairs:
        rating = ratings_by_dimension.get(pair["label"])
        if rating is None:
            raise ValueError(f"Missing rating for dimension {pair['label']!r}")

        score = float(rating["score"])
        confidence = float(rating["confidence"])
        if not 0 <= score <= 1:
            raise ValueError(f"Score out of range for {metaphor!r}: {score}")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence out of range for {metaphor!r}: {confidence}")

        normalized.append(
            {
                "dimension": pair["label"],
                "left_anchor": pair["left"],
                "right_anchor": pair["right"],
                "score": round(score, 4),
                "confidence": round(confidence, 4),
                "reasoning": str(rating.get("reasoning", "")).strip(),
            }
        )

    return {"metaphor": metaphor, "ratings": normalized}


def rate_metaphor(
    metaphor: str,
    pairs: list[dict[str, str]],
    api_key: str,
    model: str,
    max_retries: int,
    base_wait: float,
    timeout: int,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(metaphor, pairs)},
        ],
        "temperature": 0,
        "max_tokens": 1600,
        "response_format": {"type": "json_object"},
    }

    wait = base_wait
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(DEEPINFRA_URL, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 429 and attempt < max_retries:
                time.sleep(wait)
                wait *= 2
                continue
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            raw = extract_json(content)
            return normalize_result(raw, metaphor, pairs)
        except Exception as exc:
            if attempt == max_retries:
                return {"metaphor": metaphor, "error": str(exc)}
            time.sleep(wait)
            wait *= 2

    return {"metaphor": metaphor, "error": "unreachable retry state"}


def load_successful_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return results
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            metaphor = str(record.get("metaphor", "")).strip()
            if metaphor and "error" not in record:
                results[metaphor] = record
    return results


def append_jsonl(path: Path, record: dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, results: dict[str, dict[str, Any]], counts: Counter[str], pairs: list[dict[str, str]]) -> None:
    fieldnames = ["metaphor", "count"]
    for pair in pairs:
        fieldnames.extend(
            [
                f"{pair['key']}_score_0_1",
                f"{pair['key']}_confidence",
                f"{pair['key']}_reasoning",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metaphor in sorted(results):
            row: dict[str, Any] = {"metaphor": metaphor, "count": counts.get(metaphor, 0)}
            ratings = {r["dimension"]: r for r in results[metaphor].get("ratings", [])}
            for pair in pairs:
                rating = ratings.get(pair["label"], {})
                row[f"{pair['key']}_score_0_1"] = rating.get("score", "")
                row[f"{pair['key']}_confidence"] = rating.get("confidence", "")
                row[f"{pair['key']}_reasoning"] = rating.get("reasoning", "")
            writer.writerow(row)


def write_enriched_json(path: Path, source_rows: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> None:
    enriched = []
    for row in source_rows:
        out = dict(row)
        metaphor = str(row.get("metaphor", "")).strip()
        out["deepinfra_metaphor_ratings_0_1"] = results.get(metaphor, {}).get("ratings")
        enriched.append(out)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use DeepInfra LLM-as-judge to rate metaphor words on [0,1] dimensions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-enriched", type=Path, default=DEFAULT_OUT_ENRICHED)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Rate only the first N unfinished metaphors.")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--base-wait", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--include-na", action="store_true", help="Rate NA/N/A/NONE/NULL metaphor values too.")
    parser.add_argument("--dry-run", action="store_true", help="Parse inputs and report planned work without API calls.")
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    source_rows, counts = load_metaphors(args.input, skip_na=not args.include_na)
    existing = load_successful_jsonl(args.out_jsonl)
    all_metaphors = sorted(counts)
    todo = [m for m in all_metaphors if m not in existing]
    if args.limit is not None:
        todo = todo[: args.limit]

    print(f"Input rows: {len(source_rows):,}")
    print(f"Dimensions: {len(pairs):,}")
    print(f"Unique metaphor words to consider: {len(all_metaphors):,}")
    print(f"Already rated from checkpoint: {len(existing):,}")
    print(f"Remaining this run: {len(todo):,}")
    print(f"Model: {args.model}")

    if args.dry_run:
        print("Dry run only; no DeepInfra API calls made.")
        print("First unfinished metaphors:", ", ".join(todo[:20]) if todo else "(none)")
        return

    api_key = load_api_key(args.api_key_file)
    completed: dict[str, dict[str, Any]] = dict(existing)
    lock = threading.Lock()

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    rate_metaphor,
                    metaphor,
                    pairs,
                    api_key,
                    args.model,
                    args.max_retries,
                    args.base_wait,
                    args.timeout,
                ): metaphor
                for metaphor in todo
            }
            with tqdm(total=len(futures), desc="rating metaphors", unit="word") as pbar:
                for future in as_completed(futures):
                    metaphor = futures[future]
                    record = future.result()
                    append_jsonl(args.out_jsonl, record, lock)
                    if "error" not in record:
                        completed[metaphor] = record
                    pbar.update(1)

    write_csv(args.out_csv, completed, counts, pairs)
    write_enriched_json(args.out_enriched, source_rows, completed)
    print(f"Wrote checkpoint: {args.out_jsonl}")
    print(f"Wrote ratings CSV: {args.out_csv}")
    print(f"Wrote enriched JSON: {args.out_enriched}")
    print(f"Successful metaphor ratings available: {len(completed):,}/{len(all_metaphors):,}")


if __name__ == "__main__":
    main()
