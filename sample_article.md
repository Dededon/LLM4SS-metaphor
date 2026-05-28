# Task
Filter the newspaper dataset and return all articles meeting the criteria below. Preserve metadata. Do not paraphrase, summarize, or modify article text.

# Inclusion criteria
1. **Publication date**: between 1900-01-01 and 1960-12-31 (inclusive).
   - If only a year is available, include the row if 1900 ≤ year ≤ 1960.
   - Drop rows with missing/unparseable dates and log the count.

2. **Keyword match** (case-insensitive, whole-word, OR logic across the list):
   - `federal`
   - `administration`
   - `government`
   - `new deal`   ← match as a contiguous phrase, allowing 1+ whitespace between tokens
   - `the feds`   ← contiguous phrase; also match `feds` only if preceded by `the `
   - Match against the article body. Also match the headline if a headline field exists.
   - Use regex with `\b` word boundaries and the `IGNORECASE` flag. Do NOT match substrings inside other words (e.g., `governmental` should still match because of the `\b` on the left, but `federalism` should also match — confirm whether stems are wanted; default = yes, match stems via left-boundary only).

# Metadata to retain (keep every column that exists; do not drop any)
At minimum, if present in the source: `article_id`, `date`, `year`, `newspaper_name`, `region`/`state`/`city`, `column`/`section`, `page`, `headline`, `author`/`byline`, `word_count`, `url`/`source_path`, plus the full `text`.

Add two new columns:
- `matched_keywords`: list of which keywords from the list above were found.
- `match_count`: total number of keyword occurrences in the article.

# Output
- Format: one row per article, written to `filtered_articles.parquet` (fallback: `.csv` with proper quoting if parquet unavailable).
- Also write a `run_summary.json` with: total rows scanned, rows kept, rows dropped for date, rows dropped for no keyword match, per-keyword hit counts, and the date range actually observed in the output.

# Process
1. Inspect the dataset schema first (columns, dtypes, row count, a few sample rows) and report it before filtering.
2. Stream/chunk the read if the file is large; do not load the whole corpus into memory at once.
3. Apply date filter first (cheap), then keyword filter (expensive).
4. Report timing and final row count.

# Clarify before running if any of these are true
- The text field name is unclear or there are multiple candidate text columns.
- The date field is non-standard (e.g., embedded in a filename).
- I want all the samples