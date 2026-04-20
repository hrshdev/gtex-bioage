# gtex-bioage

A machine learning pipeline for estimating biological age from gene expression profiles across 54 tissues using the GTEx v11 dataset. Implements Elastic Net, Random Forest, and XGBoost baselines alongside a two-expert mixture-of-experts architecture that routes neural and non-neural tissues to specialised models.

> Research by Harson Vadivel (2020/ICT/59) — University of Vavuniya, Faculty of Applied Science, 2026.

---

## Dataset (EDA Summary)

| Property | Value |
|---|---|
| Expression matrix | 74,628 genes x 19,788 samples |
| Total subjects | 980 |
| Total samples (metadata) | 46,379 |
| Tissue types | 54 (68 including subtypes) |
| Age range | 20–79 years (decade bins) |
| Sex | 66% Male, 34% Female |
| File size | 4.31 GB parquet |

**Age distribution:**

| Age Group | Samples |
|---|---|
| 20-29 | 3,685 |
| 30-39 | 3,482 |
| 40-49 | 7,189 |
| 50-59 | 14,889 |
| 60-69 | 15,608 |
| 70-79 | 1,526 |

---

## Project Structure

```
gtex-bioage/
├── data/
│   ├── raw/                    # GTEx v11 files (not committed, see data/README.md)
│   ├── processed/              # Pipeline outputs (not committed)
│   └── README.md               # Download instructions
├── src/
│   ├── eda.py                  # Exploratory data analysis script
│   └── preprocess.py           # Chunked preprocessing pipeline
├── notebooks/
│   └── eda.ipynb               # Interactive EDA notebook
├── results/
│   └── figures/                # EDA plots
├── .gitignore
├── LICENSE
└── README.md
```

---

## Preprocessing Pipeline

The parquet file is transposed (genes=rows, samples=columns) with only 1 row group.
Chunking is done by selecting batches of **500 sample columns** at a time to stay within 8GB RAM.

| Stage | Description |
|---|---|
| 1 | Load and merge sample + subject metadata |
| 2 | Read parquet schema, match sample IDs |
| 3 | Pass 1 — compute per-gene mean, variance, expressed fraction (chunked) |
| 4 | Filter genes: mean >= 0.1, expressed in >= 10% samples, top 5,000 by variance |
| 5 | Pass 2 — build filtered (samples x 5,000 genes) matrix (chunked) |
| 6 | Save aligned metadata |

**Run:**
```bash
source ~/gtex-env/bin/activate
cd /path/to/gtex-bioage
python src/preprocess.py
```

**Outputs:**
```
data/processed/
├── expression_filtered.parquet   # samples x 5000 genes, log1p normalised
├── metadata_filtered.parquet     # aligned metadata
├── gene_list.txt                 # retained ENSG gene IDs
└── gene_stats.csv                # mean, variance, expr_frac per gene
```

---

## Roadmap

- [x] Stage 1: Exploratory data analysis
- [x] Stage 2: Chunked data preprocessing
- [ ] Stage 3: Baseline models (Elastic Net, Random Forest, XGBoost)
- [ ] Stage 4: Two-expert architecture (neural vs non-neural routing)
- [ ] Stage 5: Age gap analysis and donor metadata correlation
- [ ] Stage 6: Comparison against BiT age clock benchmark

---

## Requirements

```
python >= 3.9
pyarrow
pandas
numpy
matplotlib
scikit-learn
xgboost
jupyterlab
```

Install:
```bash
pip install pyarrow pandas numpy matplotlib scikit-learn xgboost jupyterlab
```

---

## Citation

> GTEx Consortium. (2020). The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science*, 369(6509), 1318–1330.

---

## License

MIT License — see LICENSE file.