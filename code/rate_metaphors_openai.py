"""
Rate metaphor words on bipolar scales from pairs.txt with the OpenAI API.

The script rates each unique metaphor word once, checkpoints results to JSONL,
and writes:
  - output/metaphor_word_ratings_openai.jsonl
  - output/metaphor_word_ratings_openai.csv
  - output/metaphor_results_v3_rated.json

Example:
  C:\\Users\\aruba\\Anaconda3\\envs\\agent-code\\python.exe code\\rate_metaphors_openai.py --dry-run
  $env:OPENAI_API_KEY = "..."
  C:\\Users\\aruba\\Anaconda3\\envs\\agent-code\\python.exe code\\rate_metaphors_openai.py --workers 4
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

from openai import OpenAI
from tqdm import tqdm


DEFAULT_INPUT = Path("output/metaphor_results_v3.json")
DEFAULT_PAIRS = Path("pairs.txt")
DEFAULT_OUT_JSONL = Path("output/metaphor_word_ratings_openai.jsonl")
DEFAULT_OUT_CSV = Path("output/metaphor_word_ratings_openai.csv")
DEFAULT_OUT_ENRICHED = Path("output/metaphor_results_v3_rated.json")
DEFAULT_API_KEY_FILE = Path("openai_key.txt")

SYSTEM_PROMPT = """You are an expert semantic annotator for a social-science study of metaphors.
Rate metaphor words that may describe the U.S. federal government in historical newspaper analysis.

For each bipolar pair, assign an integer score from 1 to 7:
1 means the word strongly implies the left anchor.
4 means neutral, ambiguous, or not applicable.
7 means the word strongly implies the right anchor.

Rate the general semantic implication of the metaphor word itself, not a specific article context.
If the metaphor word is unclear, malformed, or not a meaningful metaphor, use score 4 with low confidence.
Return only structured JSON matching the requested schema."""


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value or "pair"


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
            key = slug(f"{left}_{right}")
            pairs.append({"left": left, "right": right, "label": f"{left} - {right}", "key": key})
    if not pairs:
        raise ValueError(f"No rating pairs found in {path}")
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


def rating_schema(pair_labels: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "metaphor": {"type": "string"},
            "ratings": {
                "type": "array",
                "minItems": len(pair_labels),
                "maxItems": len(pair_labels),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "pair": {"type": "string", "enum": pair_labels},
                        "score": {"type": "integer", "minimum": 1, "maximum": 7},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["pair", "score", "confidence", "reasoning"],
                },
            },
        },
        "required": ["metaphor", "ratings"],
    }


def response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            value = getattr(content, "text", None)
            if value:
                parts.append(value)
    return "\n".join(parts).strip()


def build_user_prompt(metaphor: str, pairs: list[dict[str, str]]) -> str:
    pair_lines = "\n".join(f"- {p['label']}" for p in pairs)
    return f"""Metaphor word to rate: {metaphor}

Bipolar rating pairs:
{pair_lines}

Return one rating object for every pair, using the exact pair labels above."""


def normalize_result(raw: dict[str, Any], metaphor: str, pairs: list[dict[str, str]]) -> dict[str, Any]:
    pair_by_label = {p["label"]: p for p in pairs}
    ratings_by_pair = {r.get("pair"): r for r in raw.get("ratings", []) if isinstance(r, dict)}
    normalized: list[dict[str, Any]] = []
    for pair in pairs:
        rating = ratings_by_pair.get(pair["label"])
        if rating is None:
            raise ValueError(f"Missing rating for pair {pair['label']!r}")
        score = int(rating["score"])
        confidence = float(rating["confidence"])
        if not 1 <= score <= 7:
            raise ValueError(f"Score out of range for {metaphor!r}: {score}")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence out of range for {metaphor!r}: {confidence}")
        if rating["pair"] not in pair_by_label:
            raise ValueError(f"Unexpected pair label for {metaphor!r}: {rating['pair']!r}")
        normalized.append(
            {
                "pair": pair["label"],
                "left_anchor": pair["left"],
                "right_anchor": pair["right"],
                "score": score,
                "confidence": confidence,
                "reasoning": str(rating.get("reasoning", "")).strip(),
            }
        )
    return {"metaphor": metaphor, "ratings": normalized}


def rate_metaphor(
    metaphor: str,
    pairs: list[dict[str, str]],
    model: str,
    api_key: str | None,
    max_retries: int,
    base_wait: float,
) -> dict[str, Any]:
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    schema = rating_schema([p["label"] for p in pairs])
    prompt = build_user_prompt(metaphor, pairs)

    wait = base_wait
    for attempt in range(1, max_retries + 1):
        try:
            payload: dict[str, Any] = {
                "model": model,
                "instructions": SYSTEM_PROMPT,
                "input": prompt,
                "max_output_tokens": 1800,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "metaphor_word_ratings",
                        "schema": schema,
                        "strict": True,
                    }
                },
            }
            if not model.startswith("gpt-5"):
                payload["temperature"] = 0

            response = client.responses.create(**payload)
            raw = json.loads(response_text(response))
            return normalize_result(raw, metaphor, pairs)
        except Exception as exc:
            if "insufficient_quota" in str(exc):
                return {"metaphor": metaphor, "error": str(exc)}
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
                f"{pair['key']}_score",
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
            ratings = {r["pair"]: r for r in results[metaphor].get("ratings", [])}
            for pair in pairs:
                rating = ratings.get(pair["label"], {})
                row[f"{pair['key']}_score"] = rating.get("score", "")
                row[f"{pair['key']}_confidence"] = rating.get("confidence", "")
                row[f"{pair['key']}_reasoning"] = rating.get("reasoning", "")
            writer.writerow(row)


def write_enriched_json(path: Path, source_rows: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> None:
    enriched = []
    for row in source_rows:
        out = dict(row)
        metaphor = str(row.get("metaphor", "")).strip()
        out["metaphor_ratings"] = results.get(metaphor, {}).get("ratings")
        enriched.append(out)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rate metaphor words with OpenAI on pairs.txt scales.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-enriched", type=Path, default=DEFAULT_OUT_ENRICHED)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Rate only the first N unfinished metaphors.")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--base-wait", type=float, default=2.0)
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
    print(f"Rating pairs: {len(pairs):,}")
    print(f"Unique metaphor words to consider: {len(all_metaphors):,}")
    print(f"Already rated from checkpoint: {len(existing):,}")
    print(f"Remaining this run: {len(todo):,}")
    print(f"Model: {args.model}")

    if args.dry_run:
        print("Dry run only; no OpenAI API calls made.")
        print("First unfinished metaphors:", ", ".join(todo[:20]) if todo else "(none)")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and args.api_key_file.exists():
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError(
            f"OPENAI_API_KEY is not set and no API key was found in {args.api_key_file}."
        )

    lock = threading.Lock()
    completed: dict[str, dict[str, Any]] = dict(existing)

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    rate_metaphor,
                    m,
                    pairs,
                    args.model,
                    api_key,
                    args.max_retries,
                    args.base_wait,
                ): m
                for m in todo
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
