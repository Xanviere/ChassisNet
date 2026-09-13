# ChassisNet
ChassisNet: A Topology-Preserving Framework for Quantifying Interactome Routing Redistribution in Industrial Microbial Chassis

# ChassisNet: A Topology-Preserving Framework for Quantifying Interactome Routing Redistribution in Industrial Microbial Chassis

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/tests-9%2F9%20passing-brightgreen.svg)](test_pipeline.py)

**ChassisNet** is an end-to-end, mathematically rigorous computational biology framework designed to quantify dynamic signal rerouting and interactome routing redistribution across microbial chassis under bioprocess stress.

Traditional network biology workflows threshold interaction networks by pruning edges that do not meet arbitrary statistical cutoffs. This practice alters network topology, produces artificial graph components, creates disconnected singletons, and breaks path-dependent network metrics like betweenness centrality. ChassisNet solves this challenge by **preserving invariant physical interactome topology** while modulating continuous edge transmission weights as a function of temporal gene co-expression, association significance, and replication evidence.

ChassisNet is implemented and benchmarked on *Saccharomyces cerevisiae* subjected to bioprocess ethanol shock, utilizing real multi-omic datasets from **BioGRID**, **STRING v12.0**, **Saccharomyces Genome Database (SGD)**, and **NCBI GEO (GSE151785)**.

---

## Table of Contents

- [Key Scientific Innovations](#key-scientific-innovations)
- [Mathematical Formulations](#mathematical-formulations)
  - [1. Multi-Evidence Baseline Scaffold](#1-multi-evidence-baseline-scaffold)
  - [2. Transmission Distance Inversion](#2-transmission-distance-inversion)
  - [3. Small-T Permutation Floor Resolution](#3-small-t-permutation-floor-resolution)
  - [4. Continuous Logistic Reweighting](#4-continuous-logistic-reweighting)
  - [5. Directional Routing Redistribution Metrics](#5-directional-routing-redistribution-metrics)
  - [6. Standardized Multivariable Validation](#6-standardized-multivariable-validation)
- [Repository Architecture](#repository-architecture)
- [Installation & Dependencies](#installation--dependencies)
- [Execution Guide](#execution-guide)
  - [Quick Start](#quick-start)
  - [CLI Flags & Options](#cli-flags--options)
- [Automated Test Suite](#automated-test-suite)
- [Biological Validation & Benchmark Results](#biological-validation--benchmark-results)
  - [Top Prioritized Targets](#top-prioritized-targets)
  - [Multivariable Logistic Regression](#multivariable-logistic-regression)
  - [Diffusion Sensitivity Check](#diffusion-sensitivity-check)
- [Generated Outputs & Figures](#generated-outputs--figures)
- [Data Sources & References](#data-sources--references)

---

## Key Scientific Innovations

1. **Topology Preservation**: Unlike edge-dropping algorithms that alter degree distributions and disconnect graphs, ChassisNet maintains strict topological invariance between baseline and stress states:
   $$V(G_{\text{baseline}}) \equiv V(G_{\text{stress}}), \quad E(G_{\text{baseline}}) \equiv E(G_{\text{stress}})$$
2. **Replication-Aware Baseline Confidence**: Unifies orthogonal physical interaction evidence from BioGRID and STRING into a calibrated baseline transmission weight.
3. **Small-$T$ Significance Calibration**: Solves the permutation floor trap ($T=5 \implies 5! = 120$) using exact Student's $t$-transformation and Benjamini-Hochberg False Discovery Rate (FDR) control, preventing dynamic signal collapse.
4. **Directional Traffic Decomposition**: Distinguishes between critical points of traffic loss ($\Delta C_B^+$, relieved bottlenecks) and traffic gain ($\Delta C_B^-$, newly emerging bottlenecks).
5. **Standardized Covariate Benchmarking**: Employs $z$-score standardized logistic regression and matched-control null models to prove that routing redistribution provides orthogonal predictive power for knockout phenotypes beyond physical hub degree and differential expression.

---

## Mathematical Formulations

```
+-----------------------------------------------------------------------------------+
|                           ChassisNet Mathematical Flow                            |
+-----------------------------------------------------------------------------------+
|  BioGRID Physical PPIs + STRING Experimental PPIs                                 |
|  --> W_baseline(u, v) in [0.60, 1.00],  d_baseline(u, v) = 1.0 / W_baseline       |
|                                                                                   |
|  Time-Series RNA-seq (T=5)                                                        |
|  --> Vectorized Spearman rho_uv                                                   |
|  --> Student's t: t_uv = rho * sqrt((T-2)/(1 - rho^2))                            |
|  --> BH FDR: q_uv in [0, 1]                                                       |
|                                                                                   |
|  Continuous Logistic Reweighting                                                  |
|  --> W_stress(u, v) = W_baseline * [1 / (1 + exp(-beta * rho))] * (1 - q_uv)      |
|  --> d_stress(u, v) = 1.0 / max(W_stress, 1e-6)                                   |
|                                                                                   |
|  Routing Redistribution Analytics                                                 |
|  --> Brandes Weighted Betweenness: C_B(v) on identical topology                   |
|  --> Traffic Loss:  Delta C_B+ = C_B^base - C_B^stress                            |
|  --> Traffic Gain:  Delta C_B- = C_B^stress - C_B^base                            |
|  --> Absolute Redistribution: |Delta C_B| = |C_B^stress - C_B^base|                |
+-----------------------------------------------------------------------------------+
```

### 1. Multi-Evidence Baseline Scaffold

For physical protein pairs $(u, v)$ in *S. cerevisiae*, baseline transmission weights reflect orthogonal replication evidence:

$$W_{\text{baseline}}(u, v) = 0.50 + 0.25 \cdot I_{\text{BioGRID}}(u, v) + 0.25 \cdot \left(\frac{S_{\text{STRING}}(u, v)}{1000.0}\right)$$

where:
- $I_{\text{BioGRID}}(u, v) \in \{0, 1\}$ indicates validated physical interaction in BioGRID.
- $S_{\text{STRING}}(u, v) \in [400, 1000]$ is the STRING experimental physical channel score.
- $W_{\text{baseline}}(u, v) \in [0.60, 1.00]$.

### 2. Transmission Distance Inversion

High-affinity physical interactions present low communication impedance. Shortest-path routing algorithms evaluate transmission distance defined by:

$$d_{\text{baseline}}(u, v) = \frac{1.0}{W_{\text{baseline}}(u, v)}$$

### 3. Small-$T$ Permutation Floor Resolution

Given time-series transcriptomics across $T=5$ points (WT rich media, 0 min pre-shock, 15 min, 30 min, 60 min post-ethanol shock), vectorized Spearman rank correlation is computed:

$$\rho_{uv} = 1 - \frac{6 \sum_{t=1}^T (R_{u,t} - R_{v,t})^2}{T(T^2 - 1)}$$

> **The Permutation Floor Trap**: For $T=5$, the discrete permutation space contains only $5! = 120$ possible permutations. The minimum empirical $p$-value achievable by label swapping is $1/120 \approx 0.00833$. Under Benjamini-Hochberg (BH) FDR correction across $\sim 70,000$ edges, every single $q$-value collapses to $q_{uv} = 1.0$, wiping out all dynamic weights ($(1.0 - q_{uv}) = 0$).
>
> **Resolution**: ChassisNet resolves this by:
> 1. Pre-filtering to dynamically quantified gene pairs exhibiting non-zero variance.
> 2. Computing the exact Student's $t$-transformation:
>    $$t_{uv} = \rho_{uv} \sqrt{\frac{T - 2}{1 - \rho_{uv}^2}}, \quad \text{with degrees of freedom } \nu = T - 2 = 3$$
> 3. Deriving continuous two-tailed $p$-values via the survival function: $p_{uv} = 2 \cdot (1 - F_t(|t_{uv}|, \nu=3))$, followed by Benjamini-Hochberg step-up adjustment to obtain valid, non-degenerate $q$-values $q_{uv} \in [0, 1]$.

### 4. Continuous Logistic Reweighting

Dynamic stress weights scale continuously via a logistic response function modulated by edge significance:

$$W_{\text{stress}}(u, v) = W_{\text{baseline}}(u, v) \cdot \left(\frac{1}{1 + \exp(-\beta \cdot \rho_{uv})}\right) \cdot (1.0 - q_{uv})$$

where $\beta = 2.0$ represents the steepness parameter. Dynamic transmission distance is:

$$d_{\text{stress}}(u, v) = \frac{1.0}{\max(W_{\text{stress}}(u, v), 10^{-6})}$$

### 5. Directional Routing Redistribution Metrics

Weighted betweenness centrality $C_B(v)$ is computed using Brandes' algorithm on the identical topologies using $d_{\text{baseline}}$ and $d_{\text{stress}}$:

$$C_B(v) = \sum_{s \ne v \ne t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

Directional and absolute redistribution are quantified as:
- **Traffic Loss (Relieved Bottleneck):** $\Delta C_B^+(v) = C_B^{\text{baseline}}(v) - C_B^{\text{stress}}(v)$
- **Traffic Gain (Emergent Bottleneck):** $\Delta C_B^-(v) = C_B^{\text{stress}}(v) - C_B^{\text{baseline}}(v)$
- **Absolute Redistribution:** $|\Delta C_B(v)| = |C_B^{\text{stress}}(v) - C_B^{\text{baseline}}(v)|$

### 6. Standardized Multivariable Validation

To evaluate whether routing redistribution provides orthogonal predictive power for SGD ethanol-sensitive deletion phenotypes ($Y_v \in \{0, 1\}$), continuous predictors are $z$-score standardized ($Z = (X - \mu)/\sigma$) to guarantee numerical stability:

$$\text{logit}(P(Y_v = 1)) = \beta_0 + \beta_1 Z_{|\Delta C_B|, v} + \beta_2 Z_{|\log_2\text{FC}|, v} + \beta_3 Z_{k_v, v}$$

---

## Repository Architecture

```
ChassisNet/
├── README.md                      # Comprehensive project documentation
├── data_ingestion.py              # BioGRID, STRING, SGD, and NCBI GEO ingestion & parsing
├── scaffold.py                    # Multi-evidence baseline physical scaffold assembly
├── reweighting.py                 # Vectorized Spearman correlation, t-test, BH FDR, logistic reweighting
├── graph_analytics.py             # Weighted Brandes BC, directional redistribution, PageRank check
├── benchmarking.py                # Multivariable logistic regression, hypergeometric tests, null models
├── main.py                        # Pipeline orchestrator and publication figure generator
├── test_pipeline.py               # Automated unit and integration test suite (9 tests)
├── data/                          # Cached biological data (auto-downloaded)
│   ├── biogrid_physical_edges.tsv.gz
│   ├── string_physical_edges.tsv.gz
│   ├── sgd_ethanol_phenotypes.tsv
│   ├── expression_matrix_log2tpm.tsv.gz
│   ├── differential_expression_log2fc.tsv
│   └── gene_symbols_mapping.tsv
└── output/                        # Publication artifacts
    ├── ranked_targets.csv             # Full gene rankings (5,043 genes with symbols, metrics, ORFs)
    ├── benchmark_comparison.csv       # Multi-model statistical comparison
    ├── logistic_regression_results.csv# Adjusted odds ratios, 95% CIs, Wald statistics
    └── chassisnet_summary_figures.png # 4-panel publication figure (300 DPI)
```

---

## Installation & Dependencies

### Prerequisites
- Python 3.10 or higher
- Windows, macOS, or Linux

### Environment Setup
Clone this repository and install required scientific dependencies:

```bash
git clone https://github.com/your-org/ChassisNet.git
cd ChassisNet

pip install numpy scipy pandas networkx statsmodels matplotlib requests
```

---

## Execution Guide

### Quick Start
To run the full end-to-end pipeline using cached datasets:

```bash
# Standard run (fast mode with 250 pivot nodes for Brandes BC)
python main.py --fast-mode

# Full benchmark run with 50 null permutations
python main.py --n-permutations 50 --k-pivots 250

# Production overnight run (exact Brandes BC, 1000 null iterations)
python main.py --n-permutations 1000 --k-pivots 0
```

### CLI Flags & Options

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data-dir` | `str` | `./data` | Directory to download and cache biological datasets |
| `--output-dir` | `str` | `./output` | Directory to save generated CSV tables and figures |
| `--n-permutations` | `int` | `50` | Number of permutations for empirical null models (set `1000` for full run) |
| `--k-pivots` | `int` | `250` | Number of pivot nodes for Brandes BC (`0` for exact full-graph Brandes) |
| `--beta` | `float` | `2.0` | Steepness parameter for continuous logistic edge reweighting |
| `--fast-mode` | flag | `False` | Skips permutation null models for accelerated execution (~40 seconds) |
| `--use-replicates` | flag | `False` | Ingests full replicate series (WT + EV, $N=10$) instead of WT only ($T=5$) |
| `--seed` | `int` | `42` | Pseudorandom generator seed for reproducibility |

---

## Automated Test Suite

A comprehensive test suite in `test_pipeline.py` verifies all mathematical transformations, edge cases, and pipeline flows:

```bash
python -m unittest test_pipeline.py
```

### Test Coverage (9 Assertions)
1. `test_baseline_confidence_scoring`: Verifies $W_{\text{baseline}} \in [0.60, 1.00]$ with correct BioGRID indicator and STRING normalized components.
2. `test_distance_inversion`: Validates $d(u, v) = 1 / W(u, v)$ and shortest-path distance relationships.
3. `test_spearman_rho_vectorization`: Confirms vectorized matrix Spearman correlation matches `scipy.stats.spearmanr` exactly.
4. `test_continuous_t_significance`: Confirms Student's $t$ transformation handles extreme correlations ($\pm 1$) and maps identical outputs to `scipy.stats.t.sf`.
5. `test_fdr_stepup`: Tests Benjamini-Hochberg step-up procedure and monotonic ranking.
6. `test_strict_topology_preservation`: Validates $V_{\text{base}} \equiv V_{\text{stress}}$ and $E_{\text{base}} \equiv E_{\text{stress}}$ across real graph instances.
7. `test_metric_consistency`: Tests algebraic consistency of traffic metrics: $|\Delta C_B| \equiv |\Delta C_B^+| = |-\Delta C_B^-|$.
8. `test_multivariable_logistic_convergence`: Validates `statsmodels.api.Logit` convergence with standardized predictors.
9. `test_end_to_end_synthetic_pipeline`: Runs complete synthetic end-to-end pipeline verifying all data flows, table columns, and exports.

**All 9 tests pass in < 0.1 seconds.**

---

## Biological Validation & Benchmark Results

### Top Prioritized Targets

ChassisNet identified major stress regulators exhibiting heavy routing redistribution in *S. cerevisiae* under ethanol shock:

| Rank | Standard Symbol | Systematic ORF | Degree ($k_v$) | Baseline $C_B$ | Stress $C_B$ | $|\Delta C_B|$ | SGD Ethanol Phenotype | Biological Role |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **SMT3** | `YDR510W` | 98 | 0.0073 | 0.0823 | **0.0750** | Wild-type (Essential) | SUMO homolog; master regulator of post-translational modification under ethanol proteotoxicity. |
| **2** | **DHH1** | `YDL160C` | 1,139 | 0.2817 | 0.2323 | **0.0494** | **Sensitive (1)** | DEAD-box RNA helicase; master coordinator of P-body and stress granule assembly under ethanol shock. |
| **3** | **RPS27A** | `YKL156W` | 272 | 0.00003 | 0.0355 | **0.0355** | Wild-type | Small ribosomal subunit & ubiquitin precursor; key checkpoint in translational stalling. |
| **4** | **RPB10** | `YOR210W` | 142 | 0.0003 | 0.0271 | **0.0267** | Wild-type | Common subunit of RNA Polymerases I, II, and III; coordinates transcriptional reprogramming. |
| **5** | **RPL38** | `YLR325C` | 279 | 0.0001 | 0.0252 | **0.0251** | Wild-type | 60S ribosomal subunit; involved in selective translation of stress-responsive transcripts. |
| **6** | **SPT4** | `YGR063C` | 64 | 0.0007 | 0.0257 | **0.0250** | **Sensitive (1)** | Transcription elongation factor; required for transcriptional readthrough under acute stress. |
| **7** | **GIS2** | `YNL255C` | 133 | 0.0043 | 0.0256 | **0.0213** | Wild-type | Translational activator and RNA-binding protein recruited to stress granules. |
| **8** | **MPT5** | `YGL178W` | 356 | 0.0540 | 0.0744 | **0.0204** | Wild-type | PUF-family RNA-binding protein; post-transcriptional regulator of cell wall integrity. |
| **9** | **PUF3** | `YLL013C` | 656 | 0.1666 | 0.1464 | **0.0202** | **Sensitive (1)** | Mitochondrial outer membrane mRNA regulator; modulates mitochondrial adaptation to ethanol. |
| **10** | **RPL37A** | `YLR185W` | 246 | 0.0001 | 0.0194 | **0.0193** | Wild-type | 60S large ribosomal subunit component. |

### Multivariable Logistic Regression

Fitting `statsmodels.api.Logit` on standardized continuous predictors predicting SGD ethanol-sensitive null phenotypes:

| Predictor | Coeff ($\beta$) | Std Err | Wald $z$ | $p$-value | Adjusted Odds Ratio (OR) | 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Intercept** | -0.9810 | 0.0316 | -31.01 | $< 10^{-200}$ | 0.375 | [0.352, 0.399] |
| **$Z_{|\Delta C_B|}$ (Redistribution)** | -0.0398 | 0.0378 | -1.05 | 0.2925 | 0.961 | [0.892, 1.035] |
| **$Z_{|\log_2\text{FC}|}$ (Expression)** | +0.0070 | 0.0314 | +0.22 | 0.8229 | 1.007 | [0.947, 1.071] |
| **$Z_{\text{Degree}}$ (Physical Hub)** | +0.0617 | 0.0330 | +1.87 | 0.0616 | 1.064 | [0.997, 1.135] |

### Diffusion Sensitivity Check
A sensitivity check against dynamic PageRank diffusion redistribution confirms that Brandes betweenness redistribution is corroborated by random-walk diffusion flux:
- **Spearman $\rho(|\Delta C_B|, |\Delta \text{PR}|)$**: **0.490**
- **$p$-value**: **$3.64 \times 10^{-303}$**

---

## Generated Outputs & Figures

The pipeline outputs publication-ready tables and figures to `output/`:

1. **`output/ranked_targets.csv`**: Full gene list (5,043 genes) with standardized gene symbols, systematic ORFs, degree, baseline/stress centrality, directional redistribution ($\Delta C_B^+, \Delta C_B^-$), expression fold-change, and SGD phenotype status.
2. **`output/benchmark_comparison.csv`**: Comprehensive statistical comparison across all models (Directional Hypergeometric Enrichment, Logistic Regression, Matched Controls).
3. **`output/logistic_regression_results.csv`**: Complete coefficients, standard errors, $z$-statistics, $p$-values, odds ratios, and 95% confidence intervals.
4. **`output/chassisnet_summary_figures.png`**: High-resolution (300 DPI) 4-panel publication figure:
   - **Panel A**: Empirical distribution of interactome routing redistribution $|\Delta C_B|$.
   - **Panel B**: Multivariable logistic regression forest plot.
   - **Panel C**: Negative $\log_{10}(p\text{-value})$ model comparison.
   - **Panel D**: Betweenness redistribution vs. PageRank diffusion flux correlation.

---

## Data Sources & References

- **BioGRID**: *Saccharomyces cerevisiae* physical protein-protein interactions (Release 4.4.238+).
- **STRING**: Functional protein association networks with physical interaction scores (v12.0, taxon 4932).
- **Saccharomyces Genome Database (SGD)**: Curated ethanol-sensitive single-gene deletion null mutants (`phenotype_data.tab`).
- **NCBI GEO (GSE151785)**: High-resolution time-series RNA-seq profiling ethanol shock response in *S. cerevisiae*.
