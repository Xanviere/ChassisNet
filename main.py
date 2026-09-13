"""
main.py
=======
ChassisNet: A Topology-Preserving Framework for Quantifying Interactome
Routing Redistribution in Industrial Microbial Chassis.

Pipeline Orchestrator that runs the entire end-to-end workflow:
1. Ingests raw biological data (BioGRID, STRING, SGD, NCBI GEO RNA-seq).
2. Builds the replication-aware baseline physical interactome scaffold.
3. Performs continuous sigmoidal edge reweighting.
4. Computes weighted Betweenness Centrality on identical topologies.
5. Executes the 5-model benchmarking and independent multivariable validation.
6. Exports publication-ready CSV tables and summary figures.
"""

import os
import sys
import argparse
import logging
import time

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

from data_ingestion import ingest_all
from scaffold import build_physical_scaffold, get_scaffold_summary
from reweighting import reweight_physical_edges, build_identical_topology_graphs
from graph_analytics import compute_routing_redistribution
from benchmarking import run_full_benchmarking_suite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ChassisNet.Pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="ChassisNet Computational Pipeline")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="Directory to cache biological datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
        help="Directory to save generated tables and figures"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=50,
        help="Number of permutations for empirical null models (default: 50; set to 1000 for full production run)"
    )
    parser.add_argument(
        "--k-pivots",
        type=int,
        default=250,
        help="Number of pivot nodes for Brandes BC computation (default: 250; set to 0 for exact full Brandes)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Steepness parameter for continuous logistic edge reweighting (default: 2.0)"
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Run in accelerated mode (skips computationally heavy null model iterations)"
    )
    parser.add_argument(
        "--use-replicates",
        action="store_true",
        help="Use full replicate series (WT + EV, N=10) instead of single WT time series (T=5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def generate_publication_figures(
    reweighted_edges_df: pd.DataFrame,
    ranked_targets_df: pd.DataFrame,
    logistic_df: pd.DataFrame,
    enrichment_df: pd.DataFrame,
    output_path: str
):
    """
    Generates publication-ready 4-panel figure summarizing ChassisNet results:
    Panel A: Edge weight distribution (Baseline vs. Stressed).
    Panel B: Physical Degree (k_v) vs. Absolute Centrality Redistribution (|ΔC_B|).
    Panel C: Multivariable Logistic Regression Forest Plot (Odds Ratios and 95% CIs).
    Panel D: Benchmark Model Performance Comparison (-log10 p-value).
    """
    logger.info(f"Generating publication summary figures -> {output_path}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.28)

    # Panel A: Weight Distribution Comparison
    ax_a = axes[0, 0]
    w_base = reweighted_edges_df["W_baseline"].values
    w_stress = reweighted_edges_df["W_stress"].values
    
    ax_a.hist(w_base, bins=35, alpha=0.65, color="#1f77b4", label="Baseline $W_{baseline}$", density=True)
    ax_a.hist(w_stress, bins=35, alpha=0.65, color="#d62728", label="Stress $W_{stress}$", density=True)
    ax_a.set_title("A. Interactome Edge Weight Redistribution", fontsize=13, fontweight="bold", loc="left")
    ax_a.set_xlabel("Edge Transmission Weight ($W$)", fontsize=11)
    ax_a.set_ylabel("Probability Density", fontsize=11)
    ax_a.legend(frameon=True, framealpha=0.9, loc="upper right")
    ax_a.grid(True, linestyle="--", alpha=0.4)

    # Panel B: Degree vs. |ΔC_B| Scatter Plot
    ax_b = axes[0, 1]
    non_mutants = ranked_targets_df[ranked_targets_df["ethanol_phenotype"] == 0]
    mutants = ranked_targets_df[ranked_targets_df["ethanol_phenotype"] == 1]
    
    ax_b.scatter(
        non_mutants["degree"], non_mutants["abs_delta_cb"],
        alpha=0.45, s=22, color="#7f7f7f", label="Non-Phenotypic Genes"
    )
    ax_b.scatter(
        mutants["degree"], mutants["abs_delta_cb"],
        alpha=0.85, s=36, color="#d62728", edgecolors="black", linewidths=0.5,
        label="SGD Ethanol-Sensitive Mutants"
    )
    ax_b.set_title("B. Routing Redistribution vs. Physical Degree", fontsize=13, fontweight="bold", loc="left")
    ax_b.set_xlabel("Physical Degree ($k_v$)", fontsize=11)
    ax_b.set_ylabel("Absolute Centrality Redistribution ($|\\Delta C_B|$)", fontsize=11)
    ax_b.set_xscale("log")
    ax_b.legend(frameon=True, framealpha=0.9, loc="upper left")
    ax_b.grid(True, linestyle="--", alpha=0.4)

    # Annotate top 5 candidates
    top_candidates = ranked_targets_df.head(5)
    for _, row in top_candidates.iterrows():
        sym = row.get("gene_symbol", row["gene_orf"])
        label_text = f"{sym} ({row['gene_orf']})" if sym != row["gene_orf"] else sym
        ax_b.annotate(
            label_text,
            (row["degree"], row["abs_delta_cb"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=8, fontweight="bold", color="#1a1a1a"
        )

    # Panel C: Multivariable Logistic Regression Forest Plot
    ax_c = axes[1, 0]
    # Exclude intercept
    log_plot_df = logistic_df[logistic_df["predictor"] != "Intercept"].copy()
    y_pos = np.arange(len(log_plot_df))
    ors = log_plot_df["odds_ratio"].values
    ci_low = log_plot_df["ci_95_lower"].values
    ci_high = log_plot_df["ci_95_upper"].values
    pred_labels = [p.replace(" (Standardized)", "") for p in log_plot_df["predictor"]]

    ax_c.axvline(1.0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax_c.errorbar(
        ors, y_pos,
        xerr=[ors - ci_low, ci_high - ors],
        fmt="o", color="#2ca02c", ecolor="#2ca02c", elinewidth=2.2,
        capsize=5, markersize=8, capthick=1.5
    )
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(pred_labels, fontsize=10, fontweight="medium")
    ax_c.set_title("C. Independent Predictive Value (Multivariable Logistic Regression)", fontsize=13, fontweight="bold", loc="left")
    ax_c.set_xlabel("Adjusted Odds Ratio (95% CI)", fontsize=11)
    ax_c.grid(True, linestyle="--", alpha=0.4)

    # Panel D: Benchmark Model Performance Comparison
    ax_d = axes[1, 1]
    sorted_enrich = enrichment_df.sort_values(by="neg_log10_pval", ascending=True)
    y_bars = np.arange(len(sorted_enrich))
    colors = ["#1f77b4" if "Redistribution" in m else "#aec7e8" for m in sorted_enrich["model_metric"]]
    
    ax_d.barh(y_bars, sorted_enrich["neg_log10_pval"], color=colors, edgecolor="black", linewidth=0.6, height=0.6)
    ax_d.set_yticks(y_bars)
    ax_d.set_yticklabels(sorted_enrich["model_metric"], fontsize=9.5)
    ax_d.axvline(-np.log10(0.05), color="red", linestyle=":", linewidth=1.5, label="p = 0.05 significance")
    ax_d.set_title("D. Model Validation Against Gold-Standard Phenotypes", fontsize=13, fontweight="bold", loc="left")
    ax_d.set_xlabel("Enrichment Significance ($-\\log_{10} p$-value)", fontsize=11)
    ax_d.legend(frameon=True, framealpha=0.9, loc="lower right")
    ax_d.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Summary figures successfully exported.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)

    t_start = time.time()
    logger.info("================================================================================")
    logger.info("                         CHASSISNET COMPUTATIONAL PIPELINE                     ")
    logger.info("      Topology-Preserving Framework for Interactome Routing Redistribution      ")
    logger.info("================================================================================")
    logger.info(f"Configuration: Data Dir: {args.data_dir} | Output Dir: {args.output_dir}")
    logger.info(f"Parameters: Beta={args.beta}, k_pivots={args.k_pivots}, Permutations={args.n_permutations}, FastMode={args.fast_mode}")

    # Step 1: Data Ingestion
    logger.info("\n--- STEP 1: Biological Data Retrieval & Ingestion ---")
    data_dict = ingest_all(data_dir=args.data_dir)
    df_biogrid = data_dict["biogrid"]
    df_string = data_dict["string"]
    df_pheno = data_dict["phenotypes"]
    expr_df = data_dict["expression"]
    de_df = data_dict["differential_expression"]
    symbols_map = data_dict.get("symbols", {})

    # Step 2: Scaffold Construction
    logger.info("\n--- STEP 2: Multi-Tier Physical Interactome Scaffold Construction ---")
    G_baseline, scaffold_edges_df = build_physical_scaffold(df_biogrid, df_string)
    scaffold_summary = get_scaffold_summary(G_baseline)
    logger.info(f"Scaffold Summary: {scaffold_summary}")

    # Step 3: Continuous Sigmoidal Edge Reweighting
    logger.info("\n--- STEP 3: Smooth Sigmoidal Edge Reweighting Engine ---")
    reweighted_edges_df = reweight_physical_edges(
        scaffold_edges_df,
        expr_df,
        use_replicates=args.use_replicates,
        beta=args.beta
    )
    G_base_graph, G_stress_graph = build_identical_topology_graphs(reweighted_edges_df)

    # Step 4: Identical-Topology Graph Analytics
    logger.info("\n--- STEP 4: Identical-Topology Graph Centrality Analytics ---")
    k_pivots_val = None if args.k_pivots <= 0 else args.k_pivots
    redist_df, stability_meta = compute_routing_redistribution(
        G_base_graph,
        G_stress_graph,
        k_pivots=k_pivots_val,
        seed=args.seed
    )

    # Step 5: Multi-Model Benchmarking & Multivariable Validation
    logger.info("\n--- STEP 5: Multi-Model Benchmarking & Independent Validation ---")
    ranked_targets_df, enrichment_df, logistic_df, null_meta = run_full_benchmarking_suite(
        redist_df=redist_df,
        reweighted_edges_df=reweighted_edges_df,
        expr_df=expr_df,
        de_df=de_df,
        pheno_df=df_pheno,
        G_baseline=G_base_graph,
        n_permutations=args.n_permutations,
        k_pivots=k_pivots_val if k_pivots_val else 100,
        fast_mode=args.fast_mode,
        seed=args.seed
    )

    # Attach Official Gene Symbol
    ranked_targets_df["gene_symbol"] = ranked_targets_df["gene_orf"].map(
        lambda orf: symbols_map.get(orf, orf)
    )
    
    # Re-order columns with gene_symbol first
    col_order = [
        "gene_symbol", "gene_orf", "degree", "abs_delta_cb",
        "delta_cb_plus", "delta_cb_minus", "cb_baseline", "cb_stress",
        "abs_log2fc", "ethanol_phenotype", "cb_zscore",
        "predicted_log_odds_contribution", "weight_perm_pval", "temporal_null_pval"
    ]
    present_cols = [c for c in col_order if c in ranked_targets_df.columns]
    remaining_cols = [c for c in ranked_targets_df.columns if c not in present_cols]
    ranked_targets_df = ranked_targets_df[present_cols + remaining_cols]

    # Step 6: Export CSV Tables
    logger.info("\n--- STEP 6: Exporting Publication-Ready Results Tables ---")
    ranked_csv = os.path.join(args.output_dir, "ranked_targets.csv")
    ranked_targets_df.to_csv(ranked_csv, index=False)
    logger.info(f"Exported ranked gene targets ({len(ranked_targets_df):,} genes) -> {ranked_csv}")

    # Build comprehensive benchmark comparison table
    matched_data = null_meta.get("matched_control", {})
    bench_rows = []
    for _, r in enrichment_df.iterrows():
        bench_rows.append({
            "category": "Directional Overrepresentation (Hypergeometric)",
            "benchmark_model": r["model_metric"],
            "test_statistic": f"Fold Enrichment = {r['fold_enrichment']:.3f}",
            "odds_ratio": r["odds_ratio"],
            "p_value": r["hypergeom_pval"],
            "neg_log10_pval": r["neg_log10_pval"]
        })

    for _, r in logistic_df.iterrows():
        if r["predictor"] != "Intercept":
            bench_rows.append({
                "category": "Multivariable Logistic Regression (Adjusted)",
                "benchmark_model": r["predictor"],
                "test_statistic": f"Wald z = {r['z_statistic']:.3f}",
                "odds_ratio": r["odds_ratio"],
                "p_value": r["p_value"],
                "neg_log10_pval": float(-np.log10(max(r["p_value"], 1e-300)))
            })

    if matched_data:
        bench_rows.append({
            "category": "Matched-Control Empirical Null",
            "benchmark_model": "Matched Controls (k_v, mean_expr, var_expr)",
            "test_statistic": f"Relative Risk = {matched_data.get('enrichment_ratio', 1.0):.3f}",
            "odds_ratio": matched_data.get("enrichment_ratio", np.nan),
            "p_value": matched_data.get("empirical_pval", np.nan),
            "neg_log10_pval": float(-np.log10(max(matched_data.get("empirical_pval", 1.0), 1e-300)))
        })

    benchmark_summary_df = pd.DataFrame(bench_rows)
    benchmark_csv = os.path.join(args.output_dir, "benchmark_comparison.csv")
    benchmark_summary_df.to_csv(benchmark_csv, index=False)
    logger.info(f"Exported comprehensive benchmark comparison table -> {benchmark_csv}")

    logistic_csv = os.path.join(args.output_dir, "logistic_regression_results.csv")
    logistic_df.to_csv(logistic_csv, index=False)
    logger.info(f"Exported multivariable logistic regression table -> {logistic_csv}")

    # Step 7: Export Summary Figures
    logger.info("\n--- STEP 7: Rendering Publication-Quality Summary Figures ---")
    fig_path = os.path.join(args.output_dir, "chassisnet_summary_figures.png")
    generate_publication_figures(
        reweighted_edges_df=reweighted_edges_df,
        ranked_targets_df=ranked_targets_df,
        logistic_df=logistic_df,
        enrichment_df=enrichment_df,
        output_path=fig_path
    )

    elapsed = time.time() - t_start
    logger.info("\n================================================================================")
    logger.info(f"              CHASSISNET PIPELINE EXECUTION COMPLETED IN {elapsed:.1f}s         ")
    logger.info("================================================================================")
    logger.info("\nTop 10 High-Priority Engineering Targets (|Delta_C_B|):")
    top10 = ranked_targets_df[[
        "gene_orf", "degree", "abs_delta_cb", "delta_cb_plus", "delta_cb_minus",
        "abs_log2fc", "ethanol_phenotype", "predicted_log_odds_contribution"
    ]].head(10)
    print(top10.to_string(index=False))

    logger.info("\nBenchmark Overrepresentation Summary:")
    print(enrichment_df[["model_metric", "top_n", "hits_in_top", "fold_enrichment", "odds_ratio", "hypergeom_pval"]].to_string(index=False))

    logger.info("\nMultivariable Logistic Regression Summary:")
    print(logistic_df[["predictor", "coef", "odds_ratio", "z_statistic", "p_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
