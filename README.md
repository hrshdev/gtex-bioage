# gtex-bioage

Machine learning pipeline for estimating biological age from GTEx v11 gene expression across 54 tissues. Trains global baselines (Elastic Net, Random Forest, XGBoost) and a **two-expert mixture-of-experts (MoE)** model that routes neural tissues (brain, spinal cord, nerve) and non-neural tissues to separate Elastic Net experts. Benchmarks against published **REG** and **Pasta** transcriptomic clocks and interprets top clock genes and pathways.

> Research by Harson Vadivel (2020/ICT/59) — University of Vavuniya, Faculty of Applied Science, 2026.

---

## Key results (sample-stratified test set, n = 3,935)

| Model | MAE (years) | R² | Pearson r |
|---|---:|---:|---:|
| **MoE Elastic Net** | **5.51** | **0.69** | **0.83** |
| Elastic Net (global) | 5.74 | 0.67 | 0.82 |
| XGBoost | 7.32 | 0.47 | 0.70 |
| Random Forest | 9.02 | 0.21 | 0.49 |
| REG clock | 12.14 | −0.27 | 0.69 |
| Pasta clock | 45.82 | −15.44 | 0.37 |

Subject-grouped validation (no donor overlap): MoE MAE **7.03**, Elastic Net **7.24**.

---

## Dataset

### Raw GTEx v11 (download required)

| Property | Value |
|---|---|
| Expression matrix | 74,628 genes × 19,788 samples |
| Subjects | ~980 |
| Tissue types | 54 |
| Age range | 20–79 years |
| Expression file | ~4.3 GB parquet |

See [`data/README.md`](data/README.md) for download links and file names.

### Processed cohort (after QC)

| Property | Value |
|---|---|
| Samples | **19,675** |
| Genes | **5,000** (top variance, log1p TPM) |
| Filter | TPM ≥ 0.1, expressed in ≥ 10% of samples |
| Gene IDs | ENSG (version stripped) |

---

## Project structure

```
gtex-bioage/
├── data/
│   ├── raw/                         # GTEx v11 downloads (gitignored)
│   ├── processed/                   # Preprocessing outputs (gitignored)
│   └── README.md
├── models/                          # Saved MoE experts and scalers (.joblib)
├── notebooks/
│   ├── eda_preprocessed.ipynb       # EDA on processed cohort (primary)
│   └── eda_raw.ipynb                # EDA on raw metadata
├── results/
│   ├── *.csv                        # Metrics, predictions, coefficients
│   └── figures/
│       ├── raw/                     # EDA on raw metadata
│       ├── preprocessed/            # Cohort EDA (thesis Fig 1.3)
│       ├── bioage/                  # Age-gap diagnostics
│       ├── hardy/                   # Hardy-scale sensitivity plots
│       ├── clocks/                  # REG / Pasta comparison plot
│       ├── interpretation/
│       └── outliers/
├── src/
│   ├── preprocess.py                # Chunked preprocessing (low-RAM)
│   ├── data_utils.py                # Load processed matrices
│   ├── split_utils.py               # Sample- and subject-level splits
│   ├── moe_utils.py                 # Router, train/predict MoE experts
│   ├── train_baselines.py           # Elastic Net, RF, XGBoost
│   ├── train_moe_en.py              # Train two-expert MoE
│   ├── predict_moe.py               # Inference from saved models
│   ├── validate_subject_split.py    # Subject-grouped validation
│   ├── validation_utils.py          # Shared metric helpers for validation
│   ├── compare_clocks.py            # REG & Pasta benchmarks
│   ├── clock_utils.py               # External clock scoring helpers
│   ├── analyze_bioage.py            # Age-gap analysis + bioage figures
│   ├── analyze_hardy.py             # Hardy-scale confounder checks
│   ├── analyze_outliers.py          # Extreme age-gap samples
│   ├── interpret_genes.py           # Clock genes + GO/KEGG enrichment
│   ├── plot_style.py                # Shared matplotlib style for thesis plots
│   ├── eda_plot_utils.py            # Shared EDA colour / legend helpers
│   ├── plot_preprocessed_tissues.py # Regenerates preprocessed tissue figures
│   ├── plot_cohort_panels.py        # Cohort panel figures for Chapter 4
│   └── plot_results_slide.py        # Optional results-slide graphic
├── .gitignore
├── LICENSE
└── README.md
```

`src/experiments_archive/` holds superseded MoE experiments (4-expert, XGB experts). It is **not** part of the final pipeline and is **gitignored** (kept locally only).

---

## Quick start

### Requirements

- Python ≥ 3.9
- Core + plotting stack (covers training, clocks, Hardy analysis, and figure scripts):  
  `pyarrow`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `scipy`, `joblib`, `tqdm`

```bash
pip install pyarrow pandas numpy matplotlib seaborn scikit-learn xgboost scipy joblib tqdm jupyterlab
```

No extra packages are needed for `analyze_hardy.py` or the `plot_*.py` helpers.

### 1. Download data

Place GTEx v11 files in `data/raw/` — see [`data/README.md`](data/README.md).

### 2. Preprocess

Runs in chunks of 500 samples to fit ~8 GB RAM.

```bash
cd /path/to/gtex-bioage
python src/preprocess.py
```

**Outputs:** `data/processed/expression_filtered.parquet`, `metadata_filtered.parquet`, `gene_list.txt`, `gene_stats.csv`, `gene_annotation.csv`

### 3. Train and evaluate

Run from the repository root:

```bash
python src/train_baselines.py      # Global baselines
python src/train_moe_en.py         # Two-expert MoE (+ saves models/)
python src/compare_clocks.py       # REG & Pasta benchmarks
python src/analyze_bioage.py       # Age-gap CSV + figures/bioage/
python src/analyze_hardy.py        # Hardy-scale sensitivity figures
python src/analyze_outliers.py     # Outlier analysis
python src/interpret_genes.py      # Top genes + pathway enrichment
```

**Optional — subject-grouped validation** (slow, ~hours):

```bash
python src/validate_subject_split.py
```

**Optional — reload saved models without retraining:**

```bash
python src/predict_moe.py
```

**Optional — regenerate EDA / cohort figures** (processed data required):

```bash
python src/plot_preprocessed_tissues.py
python src/plot_cohort_panels.py
```

### Windows (PowerShell)

```powershell
cd C:\Research\Code\gtex-bioage
python src/preprocess.py
python src/train_moe_en.py
python src/analyze_bioage.py
```

---

## MoE architecture

```
Sample → rule-based router (tissue name)
           ├─ neural     → Elastic Net expert (brain, spinal cord, nerve)
           └─ non_neural → Elastic Net expert (all other tissues)
         → one prediction per sample (no weighted expert combination)
```

Routing logic lives in `src/moe_utils.py`. Each expert uses `ElasticNetCV` (5-fold) with `StandardScaler` preprocessing.

---

## Main outputs

| File | Description |
|---|---|
| `results/baseline_results.csv` | Global model metrics (sample split) |
| `results/moe_en_results.csv` | MoE metrics (sample split) |
| `results/moe_en_results_subject.csv` | MoE metrics (subject-grouped) |
| `results/clock_benchmark_summary.csv` | MoE vs REG vs Pasta |
| `results/test_predictions.csv` | Per-sample predictions on test set |
| `results/bioage_predictions.csv` | Predictions + age gap |
| `results/moe_gene_coefficients.csv` | Sparse expert coefficients |
| `results/top_clock_genes.csv` | Top genes per expert |
| `results/pathway_enrichment.csv` | GO/KEGG enrichment |
| `models/*.joblib` | Trained experts and scalers |

### Figures

| Folder / file | Content |
|---|---|
| `figures/raw/` | EDA on raw GTEx metadata |
| `figures/preprocessed/` | GTEx v11 processed cohort EDA |
| `figures/bioage/01–05_*.png` | Calibration, gap distribution, tissue comparison, age bias, error distribution |
| `figures/hardy/` | Hardy-scale gap and MAE plots |
| `figures/clocks/clock_comparison.png` | Model comparison bar chart |
| `figures/interpretation/` | Pathway enrichment plot |
| `figures/outliers/` | Extreme age-gap scatter plots |

---

## Pipeline status

- [x] Exploratory data analysis (raw + processed)
- [x] Chunked preprocessing (GTEx v11 → 19,675 × 5,000)
- [x] Baseline models (Elastic Net, Random Forest, XGBoost)
- [x] Two-expert Elastic Net MoE (rule-based routing)
- [x] Sample-stratified and subject-grouped validation
- [x] Biological age gap analysis
- [x] Comparison with REG and Pasta clocks
- [x] Gene and pathway interpretation

---

## Citation

> GTEx Consortium. (2020). The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science*, 369(6509), 1318–1330.

External clocks benchmarked in this repo: **REG** and **Pasta** (see `results/REG_coeffs.csv`, `results/Pasta_coeffs.csv`).

---

## License

MIT License — see [LICENSE](LICENSE).
