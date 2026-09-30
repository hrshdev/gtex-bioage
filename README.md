# gtex-bioage

Machine learning pipeline for estimating biological age from GTEx v11 gene expression across 54 tissues. Trains global baselines (Elastic Net, Random Forest, XGBoost) and a **two-expert mixture-of-experts (MoE)** model that routes neural tissues (brain, spinal cord, nerve) and non-neural tissues to separate Elastic Net experts. Benchmarks against published **REG** and **Pasta** transcriptomic clocks and interprets top clock genes and pathways.

> Research by Harson Vadivel (2020/ICT/59) — University of Vavuniya, Faculty of Applied Science, 2026.

---

## Key results (sample-stratified test set, n = 3,935)

| Model | MAE (years) | R² | Pearson r | Notes |
|---|---:|---:|---:|---|
| **MoE Elastic Net (Proposed)** | **5.51** | **0.69** | **0.83** | Two-expert neural / non-neural routing |
| Elastic Net (global) | 5.74 | 0.67 | 0.82 | Single global model across all tissues |
| XGBoost | 7.32 | 0.47 | 0.70 | GPU-accelerated gradient boosting |
| Random Forest | 9.03 | 0.21 | 0.50 | 100 trees, multi-core CPU |
| REG clock | 12.14 | −0.27 | 0.69 | Salignon et al. (2025) benchmark |
| Pasta clock | 45.82 | −15.44 | 0.37 | Published transcriptomic clock |

Subject-grouped validation (no donor overlap across folds): MoE MAE **7.03**, Elastic Net **7.24**.

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
│       ├── bioage/                  # Age-gap diagnostics (Figs 01–05)
│       ├── hardy/                   # Hardy-scale sensitivity plots
│       ├── clocks/                  # REG / Pasta comparison plot
│       ├── interpretation/          # Pathway enrichment (Thesis Fig 4.16)
│       └── outliers/                # Extreme age-gap samples
├── src/
│   ├── preprocess.py                # Chunked column preprocessing (low-RAM)
│   ├── data_utils.py                # Load processed matrices (float32 aligned)
│   ├── split_utils.py               # Sample- and subject-level splits
│   ├── moe_utils.py                 # Router, train/predict MoE experts
│   ├── train_baselines.py           # Elastic Net, RF, XGBoost (GPU supported)
│   ├── train_moe_en.py              # Train two-expert MoE (+ saves models/)
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
- Core + plotting stack:  
  `pyarrow`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `scipy`, `joblib`, `tqdm`

```bash
pip install pyarrow pandas numpy matplotlib seaborn scikit-learn xgboost scipy joblib tqdm jupyterlab
```

### 1. Download data

Place GTEx v11 files in `data/raw/` — see [`data/README.md`](data/README.md).

### 2. Preprocess

Runs in memory-safe column chunks (configurable `CHUNK_SIZE = 1000` for 12+ GB RAM / Google Colab, or `500` for 8 GB RAM) to stream through 74,628 genes without memory overflow.

```bash
cd /path/to/gtex-bioage
python src/preprocess.py
```

**Outputs:** `data/processed/expression_filtered.parquet`, `metadata_filtered.parquet`, `gene_list.txt`, `gene_stats.csv`, `gene_annotation.csv`

### 3. Train and evaluate

Run from the repository root:

```bash
python src/train_baselines.py      # Global baselines (GPU-accelerated XGBoost)
python src/train_moe_en.py         # Two-expert MoE (+ saves models/)
python src/compare_clocks.py       # REG & Pasta benchmarks
python src/analyze_bioage.py       # Age-gap CSV + figures/bioage/
python src/analyze_hardy.py        # Hardy-scale sensitivity figures
python src/analyze_outliers.py     # Outlier analysis
python src/interpret_genes.py      # Top genes + GO/KEGG pathway enrichment
```

**Optional — subject-grouped validation** (no donor overlap):

```bash
python src/validate_subject_split.py
```

**Optional — reload saved models without retraining:**

```bash
python src/predict_moe.py
```

**Optional — regenerate EDA / cohort figures:**

```bash
python src/plot_preprocessed_tissues.py
python src/plot_cohort_panels.py
```

### Google Colab

The pipeline can be executed in Google Colab connected to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Research/gtex-bioage

# Install dependencies
!pip install -q pyarrow pandas numpy matplotlib seaborn scikit-learn xgboost scipy joblib tqdm

# Execute pipeline
!python src/train_baselines.py
!python src/train_moe_en.py
!python src/compare_clocks.py
!python src/analyze_bioage.py
!python src/interpret_genes.py
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

- **Routing:** Neural tissues (Brain, Spinal Cord, Tibial Nerve; $n \approx 3{,}113$ training samples) route to the Neural Expert. All other 50 tissues ($n \approx 12{,}627$) route to the Non-Neural Expert.
- **Regularization:** Each expert uses 5-fold cross-validated `ElasticNetCV` with `StandardScaler` preprocessing across sparse regularization ratios ($L_1 \in [0.5, 1.0]$).
- **Top Clock Gene:** Identifies canonical cellular senescence marker **`CDKN2A` / *p16INK4a*** (`ENSG00000131080`) as the highest-weighted aging clock gene in both neural ($\beta = 2.29$) and non-neural ($\beta = 4.17$) experts.

---

## Main outputs

| File | Description |
|---|---|
| `results/baseline_results.csv` | Global model metrics: Elastic Net, RF, XGBoost |
| `results/moe_en_results.csv` | MoE metrics on independent test set |
| `results/moe_en_results_subject.csv` | MoE metrics (subject-grouped cross-validation) |
| `results/clock_benchmark_summary.csv` | Benchmark: MoE vs REG vs Pasta |
| `results/test_predictions.csv` | Per-sample test predictions ($n = 3{,}935$) |
| `results/bioage_predictions.csv` | Sample predictions + calculated biological age gap ($\Delta \text{Age}$) |
| `results/moe_gene_coefficients.csv` | Sparse gene weights per expert |
| `results/top_clock_genes.csv` | Top-weighted clock genes per expert |
| `results/pathway_enrichment.csv` | g:Profiler GO:BP and KEGG enrichment table |
| `models/*.joblib` | Trained expert models and scalers |

### Figures

| Folder / file | Content |
|---|---|
| `figures/raw/` | EDA on raw GTEx metadata |
| `figures/preprocessed/` | GTEx v11 processed cohort EDA (thesis Fig 1.3) |
| `figures/bioage/01–05_*.png` | Age-gap diagnostics: Calibration, Gap distribution, Tissue comparison, Age bias, Error by decade |
| `figures/hardy/` | Hardy-scale agonal state confounder check |
| `figures/clocks/clock_comparison.png` | Model comparison bar chart (MoE vs REG vs Pasta) |
| `figures/interpretation/01_pathway_enrichment.png` | GO:BP and KEGG pathway enrichment (Thesis Fig 4.16 with wrapped terms and gene count tags $n$) |
| `figures/outliers/` | Extreme age-gap sample scatter plots |

---

## Pipeline status

- [x] Exploratory data analysis (raw + processed)
- [x] Chunked preprocessing (GTEx v11 → 19,675 × 5,000)
- [x] Baseline models (Elastic Net, Random Forest, GPU-accelerated XGBoost)
- [x] Two-expert Elastic Net MoE (rule-based routing)
- [x] Sample-stratified and subject-grouped validation
- [x] Biological age gap analysis ($\Delta \text{Age}$)
- [x] Benchmark comparison against REG and Pasta clocks
- [x] Gene and pathway interpretation (Figure 4.16 legibility enhancements)

---

## Citation

> GTEx Consortium. (2020). The GTEx Consortium atlas of genetic regulatory effects across human tissues. *Science*, 369(6509), 1318–1330.

External clocks benchmarked in this repo: **REG** (Salignon et al., 2025) and **Pasta** (see `results/REG_coeffs.csv`, `results/Pasta_coeffs.csv`).

---

## License

MIT License — see [LICENSE](LICENSE).
