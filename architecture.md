# Architecture — B2B Customer Behavioral Anomaly Detection

> Living design document. Explains **why** every decision was made, not just what the code does.  
> Read top-to-bottom as a narrative, or jump to the section you need.

---

## Table of Contents

1. [Problem Framing](#1-problem-framing)
2. [Dataset](#2-dataset)
3. [Data Cleaning](#3-data-cleaning)
4. [Feature Engineering](#4-feature-engineering)
5. [Scaling & Imputation](#5-scaling--imputation)
6. [Synthetic Anomaly Injection](#6-synthetic-anomaly-injection)
7. [Model Selection](#7-model-selection)
8. [Score Ensemble](#8-score-ensemble)
9. [Evaluation Strategy](#9-evaluation-strategy)
10. [HITL Precision Simulation](#10-hitl-precision-simulation)
11. [PSI Distribution Monitoring](#11-psi-distribution-monitoring)
12. [Infrastructure & Storage](#12-infrastructure--storage)
13. [Repo Structure](#13-repo-structure)
14. [Known Limitations & Future Work](#14-known-limitations--future-work)

---

## 1. Problem Framing

### Goal

Flag B2B customers whose transaction behaviour has **deviated from their own historical pattern**. "Anomalous" is relative to the customer's baseline — not absolute. A customer who always orders large volumes is not anomalous even if their order size looks extreme compared to the population.

### Why unsupervised?

There are no labelled examples. In a real B2B context, true anomalies (fraud, churn precursors, supply chain disruptions) are:

- **Rare** — typically <5% of the customer base at any time
- **Unlabelled** — the business only confirms them retrospectively, if at all
- **Diverse** — the same root cause (e.g. financial difficulty) manifests differently across customers

Supervised learning requires a large, balanced, labelled training set. None of those conditions hold here.

### The evaluation problem

Choosing unsupervised learning creates an immediate evaluation problem: without labels, you cannot compute precision or recall on real data. This forces a **three-track evaluation strategy**:

| Track | What it measures | When it produces signal |
|---|---|---|
| Synthetic injection | Recall — can the model find anomalies we planted? | Immediately |
| HITL review loop | Precision — are the flags useful to the control team? | Over 12 weeks |
| PSI monitoring | Stability — is the input data drifting? | Each new period |

All three together give a picture of model quality without requiring pre-existing ground truth.

### Vinatex context

The original production system was built on internal Vinatex B2B transaction data using the same feature set and evaluation approach. This repo rebuilds the same system on the public UCI Online Retail II dataset, which has an almost identical schema and is close enough in domain to validate the architecture.

---

## 2. Dataset

### Source

| Property | Value |
|---|---|
| Name | UCI Online Retail II |
| URL | https://archive.ics.uci.edu/ml/datasets/Online+Retail+II |
| Source file | `data/online_retail_II.xlsx` |
| Fast format | `data/online_retail_II.parquet` |
| Sheets | Year 2009-2010, Year 2010-2011 |
| Raw rows | 1,067,371 |
| Date range | Dec 2009 — Dec 2011 |

### Why this dataset?

- **B2B e-commerce** — customers are businesses, not individual consumers. Transaction patterns are more structured and anomalies are more meaningful.
- **Customer ID present** — essential for building per-customer longitudinal profiles. Datasets without stable IDs cannot support behavioral modeling.
- **Two full years** — enough history to compute temporal features: order cadence, revenue trend, inter-order gaps.
- **Publicly available** — the work is fully reproducible without proprietary data.
- **Near-identical schema to Vinatex data** — validates that the architecture transfers between real and public datasets.

### Columns used

| Column | Type | Notes |
|---|---|---|
| `Invoice` | str | Invoice number. `C`-prefix = cancellation. |
| `StockCode` | str | Product code. Used for basket diversity. |
| `Quantity` | int | Units per line item. |
| `InvoiceDate` | datetime | Transaction timestamp. |
| `Price` | float | Unit price in GBP. |
| `Customer ID` | float → str | Excel stores as float due to nulls; cast to str. |
| `Revenue` | float | **Derived**: `Quantity × Price`. Not in source. |

### Why Parquet instead of Excel?

`pd.read_excel()` on a 1M-row `.xlsx` takes **~125 seconds** because it must parse XML, decode cell types, and handle merged cells. Parquet is a columnar binary format: the same data loads in **~0.89 seconds** — a **142× speedup**.

The `.xlsx` is kept as the source of truth and is never deleted. If the `.parquet` is missing, `load_raw()` converts automatically on first use (one-time cost, ~2 minutes). After that, all subsequent loads are fast.

```
Excel load:   125.75s
Parquet load:   0.89s
Speedup:       142×
```

**Implementation**: `src/anomaly_detection/data/loader.py` → `excel_to_parquet()`

Before writing Parquet, all `object`-typed columns are cast to `str`. This is required because the `Invoice` column contains mixed `int`/`str` values (some rows are numeric, cancellations are strings like `C536379`). Arrow (Parquet's backend) rejects mixed-type columns.

---

## 3. Data Cleaning

All rules are applied in `loader.clean()` in a fixed, deliberate order.

**Post-clean stats**: 779,425 rows (73% of raw), 5,878 unique customers.

### Rule 1 — Drop rows with no Customer ID

```python
df = df.dropna(subset=["Customer ID"])
```

**Why**: ~23% of raw rows have no Customer ID. These are likely point-of-sale or walk-in transactions with no stable identifier. Without a Customer ID, there is no way to build a longitudinal behavioral profile. These rows are fundamentally unusable for our purpose.

### Rule 2 — Remove cancellations (Invoice prefix `C`)

```python
df = df[~df["Invoice"].astype(str).str.startswith("C")]
```

**Why**: Cancellations reverse a prior transaction. Including them would corrupt per-customer aggregates — order count, total revenue, and order frequency would all be artificially inflated then deflated. The `.astype(str)` cast is required because `Invoice` is read as mixed int/str from Excel.

### Rule 3 — Remove non-positive Quantity and Price

```python
df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
```

**Why**: `Quantity ≤ 0` captures returns and manual stock adjustments. `Price ≤ 0` captures free samples, internal transfers, and data corrections. Both contaminate revenue and order-size features. Strict `> 0` (not `>= 0`) excludes zero-value records, which carry no transaction signal.

### Rule 4 — Drop exact duplicates

```python
df = df.drop_duplicates()
```

**Why**: 34,335 exact duplicate rows exist in the source — a data pipeline artefact from the original retail system (likely double-inserts). They inflate order counts and revenue without representing real transactions.

### Rule 5 — Type casts and derived column

```python
df["Customer ID"] = df["Customer ID"].astype(int).astype(str)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Revenue"]     = df["Quantity"] * df["Price"]
```

**Customer ID**: Excel stores numeric IDs with nulls as `float64` (e.g. `12345.0`). We cast to `int` first to strip the decimal, then to `str` because Customer ID is a label, not a quantity — arithmetic on it is meaningless.

**InvoiceDate**: Already parsed as datetime by `pd.read_excel` / `pd.read_parquet`, but the cast is kept explicitly for safety when loading from other sources.

**Revenue**: Always derived fresh from `Quantity × Price`. Never read from the source, which doesn't include it. Deriving it ensures consistency with the cleaning rules already applied (negative quantities and prices have been removed).

---

## 4. Feature Engineering

**Source**: `src/anomaly_detection/features/engineer.py` → `build_customer_features()`

### Design principle

Every feature must capture a **distinct dimension** of customer behaviour that could meaningfully change when something goes wrong. Redundant features (high mutual correlation) waste model capacity in distance-based models without adding signal. The correlation matrix confirms no pair exceeds |r| > 0.75.

The 11 features cover all major behavioral axes:

| Axis | Features |
|---|---|
| Frequency | `total_orders` |
| Value | `total_revenue`, `mean_order_value`, `cv_order_value` |
| Volume | `mean_order_qty`, `cv_order_qty` |
| Diversity | `mean_basket_size` |
| Recency | `recency_days` |
| Timing | `iod_mean`, `iod_std` |
| Trend | `revenue_slope` |

### The 11 features — rationale and implementation

**1. `total_orders`** — count of unique invoices  
Captures engagement frequency over the full history. A customer who normally places 50 orders per year and suddenly places 3 is anomalous even if each individual order looks normal.

**2. `total_revenue`** — sum of all invoice revenues  
The most direct proxy for customer value. A sudden revenue drop is one of the clearest early-warning signals for churn or financial difficulty.

**3. `mean_order_value`** — average revenue per invoice  
Separates high-value-per-order customers from high-frequency low-value ones. Needed to distinguish "buys a lot at once" from "buys often in small amounts".

**4. `cv_order_value`** — `std(order_value) / mean(order_value)`  
Coefficient of variation. A sudden spike in order size registers as a high CV even if the mean doesn't move much. Captures erratic spending without depending on absolute scale.  
**Edge case**: `std` is `NaN` for single-order customers (no variance observed). We set `std = 0` (consistent — no variation yet) rather than imputing the median, which would misrepresent a perfectly consistent customer as variable.

```python
cust["cv_order_value"] = cust["std_order_value"].fillna(0) / (cust["mean_order_value"] + 1e-9)
```

The `+ 1e-9` guard prevents division by zero for zero-revenue edge cases.

**5. `mean_order_qty`** — average quantity per invoice  
Volume analogue of `mean_order_value`. Necessary because price and quantity can move independently — a customer might place the same number of invoices but at much higher quantities.

**6. `cv_order_qty`** — `std(order_qty) / mean(order_qty)`  
Same coefficient-of-variation logic for quantity. Captures volume erraticism independently of value erraticism.

**7. `mean_basket_size`** — average unique SKUs per invoice  
Captures product diversity. A customer who suddenly orders only one SKU in bulk (possible stockpiling, fraud, or automated order error) registers as an abnormally low basket size, even if revenue looks normal.

**8. `recency_days`** — days since last order, relative to snapshot date  
High recency = customer has gone silent. Classic churn precursor. Computed against a **fixed snapshot date** (`max(InvoiceDate) + 1 day`) rather than "today" to keep results reproducible regardless of when the code is run.

**9. `iod_mean`** — mean inter-order gap in days  
Captures the customer's typical ordering cadence. A wholesale customer who normally orders every 14 days but shifts to every 60 days has a meaningfully different `iod_mean`.  
**Edge case**: `NaN` for single-order customers → imputed with the population median (no better estimate available for a customer we've only seen once).

**10. `iod_std`** — standard deviation of inter-order gaps  
Captures timing regularity. A customer who normally orders every 30 days but then orders 5 days apart or 200 days apart has a high `iod_std`. This detects erratic purchasing patterns independently of average cadence.  
**Edge case**: `NaN` for single-order customers → set to `0.0` (perfectly regular — only one gap observed, which is trivially consistent).

**11. `revenue_slope`** — OLS linear slope of monthly aggregated revenue  
Positive = growing account. Negative = declining account. A sudden revenue collapse registers as a sharply negative slope even if the customer is still active.  
**Implementation**: Monthly revenue is aggregated per customer, then `np.polyfit(month_index, revenue, 1)` extracts the slope. `month_index` is the number of months since the customer's first active month (not the calendar month), making it invariant to when a customer joined.  
**Edge case**: Requires `≥ 3` active months (configurable via `config.yaml → features.min_slope_months`). Customers with shorter histories → `NaN` → imputed as `0.0` (no measurable trend).

```python
monthly.groupby("Customer ID").apply(
    lambda g: _revenue_slope(g, min_slope_months), include_groups=False
)
```

The `include_groups=False` argument suppresses a pandas FutureWarning introduced in pandas 2.0 that fires when the group-by key column is accessible inside the applied function.

### Snapshot date

Recency is computed against a single fixed date (`max(InvoiceDate) + 1 day = 2011-12-10`) rather than recalculating against the current date. This makes the notebook output fully reproducible: running it today or in five years produces identical feature values.

---

## 5. Scaling & Imputation

**Source**: `src/anomaly_detection/features/engineer.py` → `BehavioralFeatureTransformer`

### SimpleImputer (strategy=`median`)

Applied as a safety net before scaling. All planned `NaN` cases are handled explicitly in `engineer.py`, but the imputer catches any edge cases a developer might introduce when adding features later (e.g. a new feature that's undefined for a particular customer cohort).

**Why median, not mean**: Several features (`total_revenue`, `recency_days`) have heavy right tails. Mean imputation would over-estimate the replacement value. Median is more robust to outliers and aligns with the "typical" customer.

### StandardScaler (zero-mean, unit-variance)

Required because features span very different scales:

| Feature | Typical range |
|---|---|
| `recency_days` | 0 – 700 |
| `total_revenue` | 0 – 500,000+ |
| `cv_order_value` | 0 – 50+ |
| `iod_std` | 0 – 200 |

Without scaling, high-magnitude features dominate distance-based models. LOF computes Euclidean distances in feature space — a difference of 100,000 in `total_revenue` dwarfs a difference of 300 in `recency_days` without scaling, even though both differences may be equally anomalous.

### Why not RobustScaler?

`RobustScaler` (median/IQR) is more resistant to outliers, but here **outliers are the signal**. A customer with extreme revenue should not be normalized "back to normal" by the scaler. `StandardScaler` preserves extreme values as genuinely large z-scores, which the anomaly detectors then correctly identify as outliers.

### Fit-on-train, transform-new

`BehavioralFeatureTransformer` fits the imputer and scaler **once** on the training population, then applies the learned parameters to any new data without refitting. This prevents data leakage when the model is applied to a new time period: the scaler's mean and variance reflect the training distribution, not the test distribution.

```python
transformer = BehavioralFeatureTransformer(min_slope_months=3)
X_scaled, customer_ids = transformer.fit_transform(df_clean)
```

The transformer returns both the scaled matrix and the corresponding customer ID list so downstream code can map scores back to customers.

---

## 6. Synthetic Anomaly Injection

**Source**: `src/anomaly_detection/evaluation/synthetic.py`

### Why inject?

Fully unsupervised models cannot be evaluated without some form of ground truth. Waiting months for the HITL loop to accumulate labels is not feasible for development iteration. Synthetic injection provides an **immediate, controllable recall estimate** during development.

### The four injection types

Each type targets a specific feature dimension, mirroring real-world B2B anomaly patterns from the Vinatex production context:

| Type | Features perturbed | Multiplier range | Real-world analogue |
|---|---|---|---|
| `volume_spike` | `mean_order_qty`, `total_revenue` | 3× – 5× | Bulk stockpiling, fraud, data entry error |
| `revenue_collapse` | `total_revenue`, `mean_order_qty` | 0.08× – 0.12× | Customer going silent, lost to competitor, financial difficulty |
| `frequency_drop` | `recency_days` | 4× – 6× | Churning customer, seasonal shutdown, operational issue |
| `timing_irregular` | `iod_std` | 5× – 10× | Erratic ordering, supply chain disruption, automated order failure |

### Design choices

**Random multipliers within range** — prevents the model from memorizing a single "canonical" anomaly value. A volume spike of exactly 4× every time would be easy to detect; a spike of 3.2× or 4.8× is more realistic.

**5% injection rate** — matches `contamination=0.05`, keeping the evaluation scenario self-consistent. Injecting 20% of customers and evaluating at a 5% threshold would produce misleading recall numbers.

**Injection on raw (unscaled) features** — the perturbation is applied before `StandardScaler`. This is correct: the injected values propagate through the same scaling transformation the real data goes through, so the resulting z-scores are realistic.

**Detectors train on injected data** — the models receive the full injected matrix and do not know which rows were perturbed. This mirrors production: in the real world, anomalies are mixed into the data stream with no labels attached.

### Limitation

Synthetic anomalies are single-feature perturbations. Real anomalies often affect multiple correlated features simultaneously — a customer in financial difficulty might show decreased revenue AND increased recency AND irregular timing all at once. Single-feature injection likely overstates how easy the detection task is, meaning our recall estimates are optimistic upper bounds.

---

## 7. Model Selection

**Source**: `src/anomaly_detection/models/detector.py`

### Why three models?

No single unsupervised detector performs best across all anomaly types. Each model has a different inductive bias. Using three models with complementary strengths reduces the risk of systematic blind spots.

### Isolation Forest

| Property | Value |
|---|---|
| `n_estimators` | 200 |
| `contamination` | 0.05 |
| `random_state` | 42 |

**Mechanism**: Randomly partitions the feature space with axis-aligned splits. Anomalies require fewer splits to isolate (shorter average path length across trees = higher anomaly score).

**Strength**: Global outliers, high-dimensional data, fast `O(n log n)`. No distance computation needed.

**Weakness**: Struggles with local density anomalies (a cluster within a cluster). Can miss anomalies that are unusual only relative to their local neighborhood.

**Why 200 trees**: Score variance decreases as the number of trees increases; beyond ~200 trees, the variance reduction is marginal for a dataset of this size (~5,878 customers). 100 trees would be noisier; 500 trees would be slower without meaningful accuracy gain.

### LOF (Local Outlier Factor)

| Property | Value |
|---|---|
| `n_neighbors` | 20 |
| `contamination` | 0.05 |

**Mechanism**: Compares the local density of a point to the densities of its k nearest neighbors. A point that is significantly less dense than its neighbors (reachability distance much larger) gets a high anomaly score.

**Strength**: Detects **local anomalies** that global methods miss. Works well when anomalies form clusters of unusual customers within a subspace.

**Weakness**: `O(n²)` naïve complexity (optimised with k-d trees but still slower than IForest). Sensitive to choice of `k`.

**Why `n_neighbors=20`**: Standard recommendation for datasets of this size. Too small (`k=5`) = noisy scores, sensitive to individual nearby points. Too large (`k=100`) = loses local structure, becomes more like a global method.

**Observed performance**: LOF achieves the best F1 of the three models on our synthetic injection test. Likely because B2B behavioral anomalies tend to be locally concentrated — customers in the same industry or size tier exhibit similar deviations, making LOF's local density comparison particularly effective.

### HBOS (Histogram-Based Outlier Score)

| Property | Value |
|---|---|
| `n_bins` | 30 |
| `contamination` | 0.05 |

**Mechanism**: Builds an independent histogram per feature. A record is anomalous if it falls in low-density bins across multiple features. Score is the sum of log-inverse bin densities.

**Strength**: `O(n)` — extremely fast. No distance computation. Not affected by the curse of dimensionality in the same way as distance-based methods.

**Weakness**: Assumes feature independence. Does not capture correlations between features. Two features may individually look normal but be jointly anomalous (e.g. high revenue + very low order count), which HBOS would miss.

**Why 30 bins**: Enough resolution to capture the shape of our feature distributions without creating sparsely populated tail bins. Fewer bins (10) would smooth over real density variation; more bins (100) would create unstable density estimates for customers in rare ranges.

### Why not OCSVM, Autoencoder, or DBSCAN?

| Model | Why rejected |
|---|---|
| **OCSVM** | Kernel methods don't scale well to 5,878 × 11 without careful kernel/C tuning. Added complexity with no clear performance advantage over IForest for tabular data. |
| **Autoencoder** | Requires architecture choices (layer sizes, activation, epochs, learning rate). Overkill for 11 features. Harder to explain to a non-technical control team ("the neural network said so" is not auditable). |
| **DBSCAN** | Assigns "noise" labels (binary) rather than a continuous anomaly score. Cannot produce a ranked list of customers by anomaly severity, which is what a control team needs to prioritize reviews. |

### Contamination = 0.05

This is a **prior**, not a measured quantity. It sets the decision threshold: PyOD flags exactly `contamination × n_customers` records as anomalies. We assume ~5% of the customer base exhibits anomalous behaviour at any time, based on the Vinatex production experience. The config makes this easy to change:

```yaml
models:
  contamination: 0.05
```

---

## 8. Score Ensemble

**Source**: `src/anomaly_detection/models/detector.py` → `AnomalyDetectorSuite.fit_predict()`

### Why ensemble?

Each model has a different sensitivity profile. A record that IForest misses (because it's a local anomaly) may be caught by LOF, and vice versa. Averaging scores reduces variance and makes the final ranking more stable.

A record flagged with high scores by all three models is very likely a true anomaly. A record flagged by only one model with borderline confidence is much less certain. The ensemble naturally encodes this.

### Min-max normalisation before averaging

Raw decision scores are on incomparable scales:

| Model | Score direction | Typical range |
|---|---|---|
| IForest | Negative (lower = more anomalous) | −0.5 to +0.5 |
| LOF | Positive (higher = more anomalous) | 1.0 to 100+ |
| HBOS | Positive log-likelihood | 0 to 50+ |

Direct averaging would let LOF dominate purely because of its magnitude. Min-max normalisation maps each model's scores to `[0, 1]` independently before averaging:

```python
lo = score_matrix.min(axis=0)
hi = score_matrix.max(axis=0)
norm = (score_matrix - lo) / (hi - lo + 1e-9)
ens_scores = norm.mean(axis=1)
```

The `+ 1e-9` guard handles the degenerate case where all scores from a model are identical.

### Threshold

The ensemble threshold is set at the `(1 - contamination)` percentile of ensemble scores:

```python
threshold = np.percentile(ens_scores, 100 * (1 - contamination))
```

This flags the same expected fraction (5%) as each individual model, keeping the evaluation scenario self-consistent.

### Why not majority voting?

Majority voting (flag if ≥ 2 of 3 models flag) was considered but produces lower recall on injected anomalies. The three models sometimes miss the same records, especially near the decision boundary — a record that all three score at 0.49 (just below threshold) gets majority-voted to "normal" even though all three models agree it's borderline. Score averaging captures this near-miss signal.

---

## 9. Evaluation Strategy

### The label scarcity problem

Standard ML evaluation (train/test split, ROC curve, confusion matrix) requires labels. We have none for real data. Three complementary signals replace them:

**Track 1 — Recall on synthetic injections**  
Plant known anomalies, measure what fraction the model recovers. Gives an immediate lower bound on recall. Computed as `recall_score(labels_true, model_labels)` where `labels_true` comes from `inject_anomalies()`.

**Track 2 — HITL precision accumulation**  
A simulated control team reviews flagged records weekly. Precision accumulates over 12 weeks. Measures whether the model's flags are actually useful to a human reviewer.

**Track 3 — PSI distribution monitoring**  
Measures whether the input feature distributions are stable between the training period and a monitoring period. An unstable distribution means the model's learned thresholds may no longer be valid.

### Metric interpretation

| Metric | Expected value | Interpretation |
|---|---|---|
| Synthetic recall | 0.13 – 0.17 | Expected for fully unsupervised at 5% contamination. Not a failure — no labels were used. |
| AUC | 0.65 – 0.72 | Meaningful lift above random (0.5) for a label-free method. |
| HITL precision (week 12) | 0.15 – 0.18 | ~1 in 6 flags confirmed as true anomaly. Useful for triage. |
| PSI (non-recency features) | < 0.10 | Stable — model reuse is safe. |

### Why not held-out validation?

Splitting the dataset temporally (train on year 1, test on year 2) would reduce the training population and destabilise behavioral features for customers with short histories. `iod_mean`, `revenue_slope`, and `iod_std` all require multiple months of data — halving the training window makes these features meaningless for many customers. Full-data training with synthetic injection is more robust.

---

## 10. HITL Precision Simulation

**Source**: `src/anomaly_detection/evaluation/hitl.py`

### What it simulates

In production, a control team reviews batches of flagged customers each week and labels each as "true anomaly" or "false alarm". This builds ground truth over time and provides a live precision estimate.

```
Config:
  hitl_weekly_sample: 20   (records reviewed per week)
  hitl_n_weeks:       12   (weeks simulated)
```

### Simulation logic

```
1. Take the set of customers flagged by the best model.
2. Each week: randomly sample up to 20 un-reviewed records.
3. Look up their synthetic label (1 = injected anomaly, 0 = normal).
4. Accumulate TP/FP counts → running precision.
```

### Design choices

**`weekly_sample=20`** — realistic for a small control team doing secondary review alongside other work. A team of 2–3 analysts reviewing ~10 records each per week.

**`n_weeks=12`** — three months of operation. Enough to see precision stabilize and for the team to build confidence (or lose it) in the model.

**Random sampling** — in practice, analysts would likely prioritize the highest-scored records first. Random sampling gives a **conservative (pessimistic) precision estimate**. If analysts work top-down by score, real precision would be higher in the early weeks.

### Why this matters

The HITL curve shows how trust in the model builds. In a governance context, control team adoption is the ultimate success metric — a model the team ignores because they don't trust it is worthless regardless of its F1 score. The simulation helps set expectations before deployment: "In week 1, expect ~5% of your flags to be confirmed. By week 12, expect ~15–18%."

---

## 11. PSI Distribution Monitoring

**Source**: `src/anomaly_detection/evaluation/psi.py`

### Formula

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

Computed over percentile-based bins derived from the baseline (expected) distribution.

### Thresholds

| PSI value | Status | Action |
|---|---|---|
| < 0.10 | STABLE | No action needed |
| 0.10 – 0.20 | SLIGHT SHIFT | Monitor closely next period |
| > 0.20 | SIGNIFICANT SHIFT | Investigate; consider retraining |

```yaml
evaluation:
  psi_slight_threshold: 0.1
  psi_significant_threshold: 0.2
```

### Why percentile-based bins?

Fixed-width bins (e.g. bin every 1,000 units of revenue) leave most bins empty for skewed distributions. Percentile-based bins ensure each bin contains a similar number of baseline observations, making the PSI estimate statistically stable even in the tails.

A `np.clip(..., 1e-4, None)` guard prevents `log(0)` when an actual bin is empty (a real event for sparse tail bins).

### Baseline vs. monitoring

We use Year 2009-2010 customers as the baseline and Year 2010-2011 as the monitoring window. In production, the baseline would be the population at the time of model training, updated quarterly.

### Expected finding: recency_days PSI ≈ 1.58

`recency_days` shows a very large PSI (~1.58) between the two years. This is **expected and not a real problem**. Recency is computed relative to a fixed snapshot date (December 2011). Year 2009-2010 customers naturally appear much older (higher recency) than Year 2010-2011 customers because the snapshot date is further from their last order.

In production, recency would be computed relative to the current date, so year-over-year PSI for recency would reflect genuine behavior changes, not an artifact of the fixed snapshot.

---

## 12. Infrastructure & Storage

### `config/config.yaml` — single source of truth

All hyperparameters, file paths, thresholds, and model settings live in one YAML file. No magic numbers in source code.

**Rationale**: A data scientist should be able to change the contamination rate, add a new injection type, or adjust the PSI threshold by editing one file — not hunting through Python modules. It also makes the effect of each parameter auditable: the config is the parameter log.

### PyOD — anomaly detection library

PyOD provides a uniform sklearn-like API (`fit` / `labels_` / `decision_scores_`) across 40+ unsupervised detectors. This means we can swap IForest for COPOD or ECOD with a one-line change in `detector.py` and no refactoring of downstream code.

### MLflow — experiment tracking

Every model run logs:
- **Params**: `contamination`, `n_features`, `n_customers`, model name
- **Metrics**: `recall`, `precision`, `f1`, `roc_auc`
- **Artifact**: serialized model object

**Rationale**: Without MLflow, comparing runs across different contamination values or feature sets requires manual bookkeeping. MLflow makes every run reproducible and auditable. The tracking URI defaults to `mlruns/` (local filesystem), so no server setup is required.

### Parquet — fast data storage

See [Section 2](#2-dataset). The `.xlsx` is kept as the source of truth and is never deleted. If the `.parquet` is missing, `load_raw()` auto-converts on first use:

```python
if path.suffix == ".parquet":
    if not path.exists():
        xlsx = path.with_suffix(".xlsx")
        excel_to_parquet(xlsx, sheets=sheets, out_path=path)
    df = pd.read_parquet(path)
```

This handles the case where the repo is cloned fresh, or where the parquet was generated on a different machine and the new machine only has the Excel file.

### `pyproject.toml` — installable package

The `src/` layout with `pyproject.toml` allows `pip install -e .` to make the package importable from anywhere without `sys.path` manipulation. The notebook falls back to `sys.path.insert(0, '../src')` for environments where it isn't installed as a package.

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

Note: `setuptools.build_meta` (not `setuptools.backends.legacy:build`) — the latter was an internal name that no longer exists in recent setuptools.

### `conda` environment

The environment is pinned in `environment.yml` and `requirements.txt`. Key version constraints:

- Python 3.11 — stable, widely supported, faster than 3.10 for numeric workloads
- pandas ≥ 2.0 — required for `include_groups=False` in groupby apply
- PyOD ≥ 1.1 — unified API for all three detectors
- `combo` — required by `pyod.models.combination` (not installed by default with PyOD)

---

## 13. Repo Structure

```
anomoly_detection/
├── config/
│   └── config.yaml                   ← all hyperparams + paths
├── data/
│   ├── online_retail_II.xlsx         ← source of truth (do not delete)
│   └── online_retail_II.parquet      ← generated once by excel_to_parquet()
├── src/
│   └── anomaly_detection/            ← installable package (pip install -e .)
│       ├── __init__.py               ← load_config() helper
│       ├── data/
│       │   └── loader.py             ← load_raw(), clean(), excel_to_parquet()
│       ├── features/
│       │   └── engineer.py           ← build_customer_features(),
│       │                                BehavioralFeatureTransformer
│       ├── models/
│       │   └── detector.py           ← AnomalyDetectorSuite, DetectionResult,
│       │                                ModelResult
│       ├── evaluation/
│       │   ├── synthetic.py          ← inject_anomalies(), InjectionResult
│       │   ├── hitl.py               ← simulate_hitl_review(), HITLResult
│       │   └── psi.py                ← compute_psi(), monitor_psi()
│       └── visualization/
│           └── plots.py              ← 7 plot functions, all auto-save to assets/
├── notebooks/
│   └── anomaly_detection_pipeline.ipynb  ← thin orchestration, no logic here
├── scripts/
│   ├── run_pipeline.py               ← CLI entry point
│   └── build_notebook.py             ← regenerates .ipynb from scratch
├── tests/
│   ├── conftest.py                   ← synthetic transaction fixtures
│   ├── test_loader.py
│   ├── test_engineer.py
│   ├── test_detector.py
│   └── test_evaluation.py
├── assets/
│   ├── pipeline.html                 ← interactive SVG pipeline diagram
│   ├── pipeline.png                  ← rendered PNG (embedded in README)
│   └── 01–07_*.png                   ← auto-saved plot outputs
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── architecture.md                   ← this file
└── README.md
```

### Design principles

1. **Config-driven** — change behaviour by editing YAML, not source code.
2. **No business logic in notebooks** — the notebook is a thin orchestration layer. All logic lives in `src/`.
3. **Single responsibility** — each module does exactly one thing. `loader.py` loads data. `engineer.py` engineers features. `detector.py` runs detectors.
4. **Testable without data** — all tests use synthetic in-memory fixtures. No Excel dependency in `tests/`. This keeps the test suite fast (<15 seconds) and CI-friendly.
5. **Fail loudly** — `_REQUIRED_COLS` check in `load_raw()` raises immediately if the dataset schema is wrong, rather than producing silent NaNs downstream.

---

## 14. Known Limitations & Future Work

### Known Limitations

**L1 — Single-period training**  
The model trains on the full two-year dataset at once. In production, it should train on a rolling 12-month window and retrain when any non-recency feature shows PSI > 0.20. Training on the full history means the model reflects 2009-2011 behavior norms, which may not match a live 2024+ dataset.

**L2 — Injection simplicity**  
Synthetic anomalies perturb one feature at a time. Real anomalies often manifest across multiple correlated features simultaneously (a fraud event might increase volume AND change timing AND reduce basket diversity). Single-feature injection likely overstates recall — the detection task is harder in practice.

**L3 — No temporal cross-validation**  
Standard cross-validation shuffles randomly, leaking future information into training for time-series data. A proper evaluation would use walk-forward validation: train on months 1–12, test on 13–14, advance window, repeat. This is not implemented.

**L4 — Contamination is an assumption**  
`contamination=0.05` is a prior, not a measured rate. If the true anomaly rate is 1% or 15%, all thresholds, precision, and recall estimates shift accordingly. Sensitivity analysis over contamination would significantly strengthen the evaluation.

**L5 — Feature independence in HBOS**  
HBOS assumes features are independent. The correlation matrix shows moderate correlations (`total_revenue` ↔ `mean_order_value`, r ≈ 0.7). HBOS underperforms as a result (lowest AUC: ~0.564). This is a structural limitation of the algorithm, not a tuning issue.

### Future Work

**F1 — COPOD / ECOD**  
Both are newer PyOD detectors based on copula/empirical CDF theory. They are parameter-free, fast, and handle feature correlations better than HBOS. Swapping them in requires a one-line change in `detector.py`.

**F2 — Temporal features**  
Revenue momentum (change in slope over the last 3 months vs. the prior 3), order acceleration, and seasonal decomposition residuals would improve detection of gradual-drift anomalies that current features miss. Customers whose decline is slow and consistent would register more clearly with these additions.

**F3 — Customer segmentation pre-stratification**  
Fitting one global model assumes all customers occupy the same behavioral space. A wholesale importer and a small retailer have fundamentally different baseline behaviors. Clustering customers first (e.g. by `total_revenue` quartile) and fitting per-segment models would improve local sensitivity without adding labeled data.

**F4 — Active learning from HITL labels**  
HITL labels are currently discarded after the simulation. In production, they should feed back into the model as weak supervision (e.g. PU learning or label propagation) to improve future runs. Even 50–100 confirmed anomaly labels per quarter would substantially improve ranking quality.

**F5 — Real-time incremental scoring**  
The current pipeline is batch: full population, once. A streaming variant would score each new invoice and update the customer's anomaly score incrementally using a sliding window of the last N transactions. This would catch anomalies within hours of occurrence rather than in the next weekly batch run.
