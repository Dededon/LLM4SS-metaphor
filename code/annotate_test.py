"""
Annotation test script — runs the first 10 rows of each sub-dataset
through DeepInfra Llama-3.1-8B and writes results to JSON-lines files.
"""

import json
import time
import requests
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY   = "Hw7UxLJSyRjvpnHedezBhhYtqTjg3Ibc"
API_URL   = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL     = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TEST_N    = 10
RETRY_MAX = 3
RETRY_WAIT = 5   # seconds between retries

DATASETS = {
    "stratified_10pct_by_year":    "stratified_10pct_by_year.parquet",
    "northeast_vs_south_regions":  "northeast_vs_south_regions.parquet",
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
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
    }

    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown code fences if model wraps output
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            result = json.loads(raw)
            # Ensure id matches the article we sent
            result["id"] = article_id
            return result

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"  [attempt {attempt}/{RETRY_MAX}] error for {article_id}: {e}")
            if attempt < RETRY_MAX:
                time.sleep(RETRY_WAIT)

    # All retries failed — return an error record
    return {"id": article_id, "reasoning": "API error after retries", "annotation": "error"}


def annotate_dataset(label: str, parquet_path: str):
    print(f"\n{'='*60}")
    print(f"Dataset: {label}")
    print(f"File:    {parquet_path}")
    print(f"Testing first {TEST_N} rows")
    print("=" * 60)

    df = pd.read_parquet(parquet_path)
    sample = df.head(TEST_N)

    results = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        article_id = str(row["article_id"])
        user_text  = build_user_message(row)
        print(f"\n[{i}/{TEST_N}] id={article_id}")
        print(f"  state={row.get('state','?')}  year={row.get('year','?')}")

        result = call_api(article_id, user_text)
        results.append(result)

        print(f"  annotation : {result.get('annotation')}")
        print(f"  reasoning  : {result.get('reasoning','')[:120]}")

    out_path = f"annotation_test_{label}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} annotations → {out_path}")

    # Quick summary
    counts = {}
    for r in results:
        counts[r.get("annotation", "error")] = counts.get(r.get("annotation", "error"), 0) + 1
    print("Summary:", counts)


if __name__ == "__main__":
    for label, path in DATASETS.items():
        annotate_dataset(label, path)

    print("\nDone.")
