## Data

Raw GTEx v11 files are **not included** in this repository (4.3 GB expression matrix + GTEx data use policy). Download them locally, then run `python src/preprocess.py` from the repo root.

---

### Download instructions

**Step 1 — Register at the GTEx portal:**  
https://gtexportal.org

**Step 2 — Download these 3 files:**

| File | Section | Size |
|---|---|---|
| `GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet` | Bulk Tissue Expression | ~4.3 GB |
| `GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt` | Annotations | ~37 MB |
| `GTEx_Analysis_v11_Annotations_SubjectPhenotypesDS.txt` | Annotations | ~20 KB |

Direct link: https://gtexportal.org/home/downloads/adult-gtex/overview

**Step 3 — Place files here:**

```
data/raw/
├── GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm.parquet
├── GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt
└── GTEx_Analysis_v11_Annotations_SubjectPhenotypesDS.txt
```

---

### Raw data structure

**Expression parquet**
- 74,628 genes (rows) × 19,790 columns
- Columns: `Description`, [19,788 sample IDs], `Name`
- `Name`: versioned ENSG IDs (e.g. `ENSG00000290825.2`)
- Values: TPM (transcripts per million)
- Layout: genes = rows, samples = columns (transposed vs typical expression matrices)

**Sample attributes** — ~48k rows, per-sample QC and tissue labels (`SAMPID`, `SMTSD`, `SMRIN`, …)

**Subject phenotypes** — ~981 donors: `SUBJID`, `SEX`, `AGE`, `DTHHRDY` (Hardy scale)

---

### Preprocessing (`src/preprocess.py`)

The script merges metadata, keeps samples present in both metadata and the expression file, and applies QC filters before building the modelling matrix.

| Step | Rule |
|---|---|
| Metadata merge | Sample attributes + subject phenotypes on `SUBJID` |
| Required fields | Drop samples missing tissue (`SMTSD`), age, sex, or Hardy scale |
| Hardy scale | Keep `DTHHRDY` in {0, 1, 2, 3, 4} |
| Expression match | Keep only samples with TPM columns in the parquet |
| Gene filter | Mean TPM ≥ 0.1, expressed in ≥ 10% of retained samples |
| Feature set | Top **5,000** genes by variance |
| Transform | log1p(TPM) |
| Gene IDs | Strip Ensembl version suffix → `ENSG########` |

Chunking reads **500 sample columns** at a time so the pipeline runs on ~8 GB RAM.

**Run:**

```bash
cd /path/to/gtex-bioage
python src/preprocess.py
```

---

### Processed outputs

After preprocessing, expect **19,675 samples × 5,000 genes** (post-QC cohort used in this study).

```
data/processed/
├── expression_filtered.parquet   # samples × 5,000 genes, log1p TPM
├── metadata_filtered.parquet     # aligned metadata (index = SAMPID, includes AGE_MID)
├── gene_list.txt                 # retained ENSG gene IDs
├── gene_stats.csv                # per-gene mean, variance, expr_frac
└── gene_annotation.csv           # ENSG, gene_symbol, name_versioned + stats
```

| Output | Shape / notes |
|---|---|
| `expression_filtered.parquet` | 19,675 × 5,000 |
| `metadata_filtered.parquet` | 19,675 rows; columns include `SMTSD`, `AGE`, `AGE_MID`, `SEX`, `DTHHRDY`, `SMRIN` |
| `gene_list.txt` | 5,000 ENSG IDs |
| `gene_annotation.csv` | Used by `src/interpret_genes.py` for gene symbols |

These paths are **gitignored** — regenerate locally after downloading raw data.

**Regenerate annotation only** (if `gene_list.txt` and `gene_stats.csv` already exist):

```bash
python src/preprocess.py --annotation-only
```

---

### Processed cohort summary

| Property | Value |
|---|---|
| Samples | 19,675 |
| Genes | 5,000 |
| Tissues | 54 |
| Donors | ~980 |
| Age range | 20–79 years (decade bins) |

EDA figures for this cohort are in `results/figures/preprocessed/` (from `notebooks/eda_preprocessed.ipynb`).
