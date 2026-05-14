# CS6370 NLP — Information Retrieval Project

**Course:** CS6370 Natural Language Processing, IIT Madras  
**Dataset:** Cranfield collection (1 400 documents, 225 queries)

---

## Overview

This project builds and iteratively improves an Information Retrieval (IR) system evaluated on the Cranfield aeronautics corpus. Work is split into five parts:

| Part | Description | Entry point |
|------|-------------|-------------|
| 1 | Toy IR system (theory) + preprocessing pipeline implementation | `main.py` |
| 2 | TF-IDF Vector Space Model (VSM) retrieval | `src/informationRetrieval.py` |
| 3 | Evaluation metrics (Precision, Recall, F-score, AP, nDCG, MAP, MRR) | `src/evaluation.py` |
| 4 | Failure-mode analysis of the VSM baseline | `scripts/part4_analysis.py` |
| 5 | Improved retrieval systems: BM25, LSA, WordNet Query Expansion | `scripts/part5.py` |

---

## Directory Structure

```
NLP_Project/
├── main.py                         # Preprocessing pipeline entry point
├── cranfield/                      # Cranfield dataset
│   ├── cran_docs.json              # 1 400 documents
│   ├── cran_queries.json           # 225 queries
│   └── cran_qrels.json             # Relevance judgements (1–4 scale)
├── src/                            # Core library modules
│   ├── sentenceSegmentation.py     # Naive and Punkt segmenters
│   ├── tokenization.py             # Naive and Penn Treebank tokenizers
│   ├── inflectionReduction.py      # Porter stemmer + WordNet lemmatizer
│   ├── stopwordRemoval.py          # NLTK-based stopword removal
│   ├── util.py                     # Shared utilities
│   ├── informationRetrieval.py     # TF-IDF VSM index and ranking
│   └── evaluation.py               # All evaluation metrics
├── scripts/                        # Analysis and runner scripts
│   ├── part4_analysis.py           # VSM diagnostics → results/part4/
│   ├── part5.py                    # BM25 / LSA / QE evaluation → results/part5/
│   ├── failure_compare.py          # Per-query comparison across all four systems
│   ├── analyze_inflection.py       # Inflection-reduction vocabulary analysis
│   ├── mode1_candidates.py         # Queries failing due to synonymy gaps
│   ├── mode2_candidates.py         # Queries failing due to negation / focus drift
│   └── mode3_candidates.py         # Queries failing due to OOV tokens
├── results/                        # Generated outputs (not committed)
│   ├── preprocessing/              # Segmented / tokenized / stemmed text files
│   ├── part4/                      # Aggregate metrics, per-query diagnostics
│   └── part5/                      # JSON results, evaluation tables, figures
└── docs/
    ├── part1/                      # Part 1 LaTeX report
    ├── part4.tex                   # Part 4 failure-analysis report
    └── spec/                       # Original assignment spec and LaTeX template
```

---

## Setup

```bash
pip install nltk spacy numpy scipy matplotlib

python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
"

python -m spacy download en_core_web_sm
```

---

## How to Run

### Part 1 — Preprocessing pipeline

```bash
# Default: punkt segmenter, Penn Treebank tokenizer, output to results/preprocessing/
python main.py -dataset cranfield

# Customise segmenter / tokenizer
python main.py -dataset cranfield -segmenter naive -tokenizer naive

# Interactive custom query
python main.py -dataset cranfield -custom
```

### Part 4 — VSM failure analysis

```bash
python scripts/part4_analysis.py
# Outputs: results/part4/{aggregate_metrics.csv, per_query_diagnostics.csv,
#                          failure_cases.txt, oov_summary.txt}

# Inspect a specific query
python scripts/part4_analysis.py --query 22

# Run mode-specific candidate finders (from project root)
python scripts/mode1_candidates.py   # synonymy gaps
python scripts/mode2_candidates.py   # negation / focus drift
python scripts/mode3_candidates.py   # OOV tokens
```

### Part 5 — Advanced retrieval systems

```bash
python scripts/part5.py
# Outputs: results/part5/{results.json, figures/, evaluation_tables.tex}

# Compare systems on specific query IDs
python scripts/failure_compare.py 1 6 13 14
```

> **Note:** All scripts must be run from the project root directory so that relative paths (`cranfield/`, `results/`) resolve correctly.

---

## Results Summary

All numbers are averaged over 225 Cranfield queries at k = 10.

| System | MAP | MRR | nDCG@10 | P@10 |
|--------|-----|-----|---------|------|
| VSM (TF-IDF baseline) | 0.369 | 0.740 | 0.464 | 0.284 |
| **BM25** | **0.401** | **0.799** | **0.493** | **0.289** |
| LSA (k = 300) | 0.346 | 0.713 | 0.436 | 0.264 |
| WordNet Query Expansion | 0.342 | 0.714 | 0.434 | 0.260 |

**Key findings:**
- BM25 improves MAP by +8.7% over VSM (p = 4.1 × 10⁻⁸, Wilcoxon signed-rank test).
- LSA and WordNet QE hurt aggregate metrics despite partially addressing synonymy failures — the latent space gains on Mode-1 queries but loses on simpler ones.
- 46 / 225 queries (20.4 %) contain OOV tokens after preprocessing; two documents (IDs 471, 995) become entirely unreachable after stopword removal.

---

## Dataset

The **Cranfield collection** is a standard IR benchmark from aeronautics literature:

| File | Content |
|------|---------|
| `cran_docs.json` | 1 400 documents with `id`, `title`, `author`, `body` fields |
| `cran_queries.json` | 225 queries with `query_num` and `query` fields |
| `cran_qrels.json` | Relevance judgements: scores 1 (highly relevant) – 4 (marginally relevant) |

Original dataset: [University of Glasgow IR resources](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/).
