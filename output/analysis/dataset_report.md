---
editor_options: 
  markdown: 
    wrap: 72
---

# Dataset Preparation & Annotation Report

**Project:** Regional Framing of the US Federal Government in Local
Newspapers, 1920–1940\
**Prepared by:** Tingyu Dai\
**Date:** May 2026

------------------------------------------------------------------------

## 1. Dataset Narrowing — Rationale

The original dataset (`filtered_articles_1920_1940.parquet`) contains
**1,253,905** articles from US local newspapers spanning 1920–1940. Two
sub-datasets were derived to serve different analytical purposes.

### Dataset 1 — Stratified 10% Sample by Year (`stratified_10pct_by_year`)

A 10% random sample was drawn within each year using stratified sampling
(`random_state=42` for reproducibility). This preserves the exact
temporal distribution of the original corpus and is suitable for
longitudinal analysis that does not require geographic granularity.

### Dataset 2 — Northeast vs. South Regional Comparison (`northeast_vs_south_regions`)

This dataset isolates two regions with historically divergent attitudes
toward federal power, enabling direct hypothesis testing of the
**location hypothesis** (H2). The classification follows Civil War-era
political alignment as the clearest historical proxy for the
states'-rights vs. federal-power ideological divide.

#### Northeast (7 states, 249,513 articles)

Defined as New England and core Mid-Atlantic states — the historical
heartland of Hamiltonian federalism, strong federal institutions, and
pro-Union politics.

| State         | Articles |
|---------------|----------|
| Connecticut   | 155,387  |
| New York      | 88,992   |
| Maine         | 2,653    |
| New Jersey    | 1,957    |
| Pennsylvania  | 424      |
| Massachusetts | 84       |
| Vermont       | 16       |

#### South (8 states, 83,097 articles)

Defined as the former Confederate states present in the corpus — the
historical core of states'-rights ideology, consistent opposition to
federal expansion, and "anti-big-government" political culture.

| State          | Articles |
|----------------|----------|
| Alabama        | 47,421   |
| Arkansas       | 17,283   |
| North Carolina | 9,400    |
| Georgia        | 2,649    |
| Florida        | 2,549    |
| Mississippi    | 2,031    |
| Louisiana      | 1,739    |
| South Carolina | 25       |

#### States Deliberately Excluded and Why

| Excluded | Reason |
|----|----|
| **District of Columbia** | Seat of the federal government; DC newspapers cover federal affairs from an insider/proximity perspective, not a regional ideological one — including it would severely bias the Northeast group |
| **Delaware & Maryland** | Border slave states that did not secede; politically ambiguous, and including them would dilute the ideological contrast between regions |
| **West Virginia** | Formed precisely by breaking away from Virginia *to remain in the Union*; constitutionally and politically anti-Confederate, despite geographic proximity to the South |
| **Kentucky** | Never formally seceded; maintained a Union-loyal state government throughout the Civil War; too ambiguous to assign cleanly to either ideological camp |
| All Midwest/Western states | Outside the scope of the Northeast–South comparison |

The rationale for strict exclusion of border states is that the research
hypotheses hinge on a clean ideological contrast. Including ambiguous
cases would reduce statistical power and weaken the interpretability of
any regional differences found.

------------------------------------------------------------------------

## 2. Dataset Structure and Basic Information

Both datasets retain the full schema of the original corpus, with
Dataset 2 adding one new column (`region`).

### Shared Column Schema

| Column | Type | Description |
|----|----|----|
| `article_id` | string | Unique article identifier |
| `newspaper_name` | string | Name of the publication |
| `edition` | string | Edition number |
| `date` | string | Publication date (YYYY-MM-DD) |
| `year` | int | Publication year |
| `page` | string | Page number |
| `headline` | string | Article headline |
| `byline` | string | Author byline (often empty) |
| `article` | string | Full article text |
| `word_count` | int | Word count of article |
| `state` | string | State of publication |
| `matched_keywords` | string | Keywords that triggered inclusion in original filter |
| `match_count` | int | Number of keyword matches |

### Dataset 1 — `stratified_10pct_by_year.parquet`

| Attribute          | Value          |
|--------------------|----------------|
| Total articles     | 125,390        |
| Years covered      | 1920–1940      |
| States covered     | 39 states + DC |
| Average word count | \~278 words    |
| File size          | 142 MB         |

### Dataset 2 — `northeast_vs_south_regions.parquet`

| Attribute          | Value                                 |
|--------------------|---------------------------------------|
| Total articles     | 332,610                               |
| Years covered      | 1920–1940                             |
| Regions            | Northeast (249,513) / South (83,097)  |
| States covered     | 15 states across 2 regions            |
| Average word count | \~276 words                           |
| File size          | 363 MB                                |
| Additional column  | `region` — `"Northeast"` or `"South"` |

------------------------------------------------------------------------

## 3. Automated Annotation

Each article (headline + full text) was passed to **DeepInfra's
Llama-3.1-8B-Instruct** model via API, with a structured prompt asking
the model to judge whether the article contains a *meaningful mention of
the US federal government* — explicitly excluding state, local, and
foreign governments. The model returned a three-way label (`yes` / `no`
/ `not sure`) with a one-to-two sentence reasoning trace for each
decision.

### Annotation Output Schema

Each annotation is stored as one JSON object per line (`.jsonl` format):

``` json
{
  "id": "<article_id>",
  "reasoning": "The article mentions the House war expenditures committee and federal relief legislation, indicating a direct reference to the US federal government.",
  "annotation": "yes"
}
```

| Field        | Type   | Values                                              |
|--------------|--------|-----------------------------------------------------|
| `id`         | string | Matches `article_id` in the parquet file            |
| `reasoning`  | string | 1–2 sentence explanation of the annotation decision |
| `annotation` | string | `"yes"` / `"no"` / `"not sure"`                     |

### Annotation Results Summary

**Dataset 1 — Stratified**

| Label    | Count   | \%    |
|----------|---------|-------|
| yes      | 100,588 | 80.2% |
| no       | 19,267  | 15.4% |
| not sure | 5,368   | 4.3%  |
| error    | 167     | 0.1%  |

**Dataset 2 — Regional**

| Label    | Northeast | \%    | South  | \%    |
|----------|-----------|-------|--------|-------|
| yes      | 186,650   | 74.9% | 65,631 | 78.9% |
| no       | 49,346    | 19.8% | 13,310 | 16.0% |
| not sure | 13,065    | 5.2%  | 4,013  | 4.8%  |
| error    | 452       | 0.2%  | 142    | 0.2%  |

**Temporal trend of `yes` rate (Stratified dataset):**\
The proportion of articles mentioning the federal government rises
noticeably in 1932–1934 (peaking at \~87%), coinciding with the onset of
the Great Depression and the rollout of the New Deal. This is consistent
with H1 (the event hypothesis) and suggests the automated annotation
captures a real signal.

------------------------------------------------------------------------

## 4. Recommendations for Next Steps

### 4.1 Which Research Questions Each Dataset Is Best Suited For

**Use the Stratified dataset (`stratified_10pct_by_year`) for:** -
**Longitudinal / temporal analysis** — testing H1 (the event
hypothesis): whether the Great Depression and New Deal changed how
newspapers framed the federal government, across all states and
regions - **National-level trend modeling** — the broad geographic
coverage (39 states) makes it suitable for corpus-wide time-series
analysis - **Computational pilot work** — at 125K articles it is more
manageable for iterative model development and prompt testing

**Use the Regional dataset (`northeast_vs_south_regions`) for:** -
**Regional comparison** — testing H2 (the location hypothesis): whether
Southern newspapers consistently framed the federal government
differently from Northeastern ones - **Interaction of event × location**
— did the Great Depression *differentially* shift framing in the South
vs. the Northeast? - **Metaphor analysis** — if the core analysis is
about specific metaphors used to frame federal programs, this dataset
provides a focused, theory-motivated corpus where regional contrasts
should be most visible

### 4.2 Recommendations for Human Annotation

The automated annotation provides a coarse filter (`yes` / `no` /
`not sure`) indicating federal government relevance. The next layer of
annotation — presumably coding for *specific framing types or metaphors*
— should be done by humans. Suggested approach:

1.  **Scope:** Focus human annotation on the `yes`-labeled articles
    only. This reduces the annotation pool from 332,610 to \~252,000
    (regional) or \~100,000 (stratified) — a substantial reduction that
    eliminates clearly irrelevant content.

2.  **Handle `not sure` separately:** The \~5% `not sure` articles
    represent genuinely ambiguous cases (e.g., "the government
    announced" without specifying which level). Consider having one
    annotator review a random sample of these to determine whether they
    warrant inclusion before discarding them.

3.  **Validate the automated annotation first:** Before committing to
    human annotation at scale, draw a random sample of \~200 articles
    across all three labels (`yes`, `no`, `not sure`) and have one team
    member verify the model's decisions. A target inter-rater agreement
    (model vs. human) of ≥85% would confirm the automated filter is
    reliable enough to use as a first pass.

4.  **Annotation unit:** For framing/metaphor coding, the article is
    likely too long as a single unit. Consider annotating at the
    *paragraph* or *sentence* level for articles flagged `yes`, which
    will yield richer and more precise framing codes.

5.  **Codebook anchoring:** Given the 1920–1940 historical context, the
    codebook should anticipate recurring federal programs (e.g., CCC,
    WPA, AAA, Federal Reserve actions) and branch-specific references
    (congressional debates, presidential addresses, Supreme Court
    rulings on New Deal legislation) as likely framing occasions.

------------------------------------------------------------------------

*Annotation files:*\
- `annotations_stratified_10pct_by_year.jsonl` (125,390 records)\
- `annotations_northeast_vs_south_regions.jsonl` (332,610 records)
