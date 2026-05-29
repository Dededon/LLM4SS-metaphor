"""
Full annotation run — DeepInfra Llama-3.1-8B
Supports concurrent requests, checkpoint/resume, and progress bar.

Usage:
    python3 annotate_full.py                     # run both datasets
    python3 annotate_full.py --workers 50        # override concurrency
    python3 annotate_full.py --dataset regional  # run one dataset only
"""

import argparse
import json
import os
import time
import threading
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY    = "Hw7UxLJSyRjvpnHedezBhhYtqTjg3Ibc"
API_URL    = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL      = "meta-llama/Meta-Llama-3.1-8B-Instruct"
WORKERS    = 50
RETRY_MAX  = 4
BASE_WAIT  = 5       # seconds before first retry; doubles each attempt

DATASETS = {
    "stratified": (
        "stratified_10pct_by_year.parquet",
        "annotations_stratified_10pct_by_year.jsonl",
    ),
    "regional": (
        "northeast_vs_south_regions.parquet",
        "annotations_northeast_vs_south_regions.jsonl",
    ),
}

SYSTEM_PROMPT = """\
Role: You are an expert annotator with deep knowledge of US newspapers and the federal government.
Task: You will be given a historical US newspaper article (headline + full text). Your task is to annotate whether the article contains any meaningful mention of the US federal government, including federal agencies, federal programs, or any branch of the federal government (executive, legislative, or judicial).
Context: The newspapers are US publications from 1920 to 1940. Focus exclusively on the US federal government — do not annotate for state, local, or foreign governments.
What qualifies as a meaningful mention:
Any reference to federal branches (Congress, the President, federal courts)
Any reference to federal agencies (e.g., the FDA, FBI, Treasury)
Any reference to federal programs or legislation (e.g., a federal relief program, a congressional bill)
Does not qualify: passing use of the word "government" with no clear federal reference, or articles solely about state/local/foreign governments
Output: Annotate using one of three labels:
yes — the article contains a meaningful mention of the US federal government
no — it does not
not sure — the content is ambiguous (e.g., unclear whether the government reference is federal or state)
Output Format:
{
  "id": "<pre-existing article ID>",
  "reasoning": "<1–2 sentences explaining your decision>",
  "annotation": "yes / no / not sure"
}
Additional instructions:
Use your best judgment when article text is short or incomplete
Return only the JSON object, no additional commentary\
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_user_message(row: pd.Series) -> str:
    headline = str(row.get("headline", "")).strip()
    article  = str(row.get("article",  "")).strip()
    parts = []
    if headline and headline.lower() not in ("nan", "none", ""):
        parts.append(f"Headline: {headline}")
    if article and article.lower() not in ("nan", "none", ""):
        parts.append(f"Article:\n{article}")
    return "\n\n".join(parts) if parts else "(no text)"


def call_api(article_id: str, user_text: str) -> dict:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":       MODEL,
        "messages":    [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        "temperature": 0.0,
        "max_tokens":  256,
    }

    wait = BASE_WAIT
    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)

            if resp.status_code == 429:
                time.sleep(wait)
                wait *= 2
                continue

            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            result = json.loads(raw)
            result["id"] = article_id
            return result

        except (requests.RequestException, json.JSONDecodeError, KeyError):
            if attempt < RETRY_MAX:
                time.sleep(wait)
                wait *= 2

    return {"id": article_id, "reasoning": "API error after retries", "annotation": "error"}


# ── Per-dataset runner ────────────────────────────────────────────────────────

def load_done_ids(out_path: str) -> set:
    """Return the set of article IDs already written to the output file."""
    if not os.path.exists(out_path):
        return set()
    done = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def run_dataset(label: str, parquet_path: str, out_path: str, workers: int):
    print(f"\n{'='*64}")
    print(f"Dataset  : {label}")
    print(f"Input    : {parquet_path}")
    print(f"Output   : {out_path}")
    print(f"Workers  : {workers}")

    df = pd.read_parquet(parquet_path)
    total = len(df)

    done_ids = load_done_ids(out_path)
    if done_ids:
        print(f"Resuming : {len(done_ids):,} already done, {total - len(done_ids):,} remaining")

    todo = df[~df["article_id"].astype(str).isin(done_ids)]
    if todo.empty:
        print("All rows already annotated — skipping.")
        return

    write_lock = threading.Lock()

    def process(row):
        article_id = str(row["article_id"])
        user_text  = build_user_message(row)
        result     = call_api(article_id, user_text)
        with write_lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    rows    = [row for _, row in todo.iterrows()]
    counts  = {}
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, row): row for row in rows}
        with tqdm(total=len(rows), desc=label, unit="art") as pbar:
            for future in as_completed(futures):
                result = future.result()
                ann = result.get("annotation", "error")
                counts[ann] = counts.get(ann, 0) + 1
                pbar.update(1)
                pbar.set_postfix(
                    yes=counts.get("yes", 0),
                    no=counts.get("no", 0),
                    unsure=counts.get("not sure", 0),
                    err=counts.get("error", 0),
                )

    elapsed = time.time() - t_start
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    print(f"\nFinished {len(rows):,} rows in {h}h {m}m {s}s")
    print(f"Results  : {counts}")
    print(f"Saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"Concurrent API calls (default {WORKERS})")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()) + ["all"],
                        default="all", help="Which dataset to run (default: all)")
    args = parser.parse_args()

    targets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}

    total_rows = sum(
        len(pd.read_parquet(p)) - len(load_done_ids(o))
        for _, (p, o) in targets.items()
    )
    est_secs = total_rows * 1.61 / args.workers
    h, rem   = divmod(int(est_secs), 3600)
    m        = rem // 60

    # Cost estimate: DeepInfra Llama-3.1-8B $0.055/1M tokens
    avg_input_tokens  = 825   # sys prompt (~450) + headline (~15) + article (~360)
    avg_output_tokens = 100
    cost = total_rows * (avg_input_tokens + avg_output_tokens) / 1e6 * 0.055

    print(f"\nRows to annotate     : {total_rows:,}")
    print(f"Estimated runtime    : ~{h}h {m}m  ({args.workers} workers, 1.61s/call avg)")
    print(f"Estimated API cost   : ~${cost:.2f}  (DeepInfra $0.055/1M tokens)")
    print("Note: runtime may vary ±20% depending on article length and API load\n")

    for label, (parquet, out) in targets.items():
        run_dataset(label, parquet, out, args.workers)

    print("\nAll done.")


if __name__ == "__main__":
    main()
