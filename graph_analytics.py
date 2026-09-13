"""
graph_analytics.py
==================
Computes weighted Betweenness Centrality on identical topologies using Brandes' algorithm:
1. G_baseline with distance attribute d_baseline.
2. G_stress with distance attribute d_stress.
3. Quantifies interactome routing redistribution metrics:
   - Traffic Loss:           ΔC_B+(v) = C_B,baseline(v) - C_B,stress(v)
   - Traffic Gain:           ΔC_B-(v) = C_B,stress(v) - C_B,baseline(v)
   - Absolute Redistribution: |ΔC_B(v)| = |C_B,baseline(v) - C_B,stress(v)|
4. Sensitivity & Stability Check:
   - Computes PageRank redistribution (|ΔPR|) and rank correlation against |ΔC_B|.
   - Optional Current-Flow Betweenness on candidate subgraphs.
"""

import logging
from typing import Dict, Tuple, Optional, Any

import pandas as pd
import numpy as np
from scipy import stats
import networkx as nx

logger = logging.getLogger("ChassisNet.Analytics")


def compute_weighted_betweenness_centrality(
    G: nx.Graph,
    weight_attr: str = "distance",
    normalized: bool = True,
    k_pivots: Optional[int] = None,
    seed: Optional[int] = 42
) -> Dict[str, float]:
    """
    Computes weighted Betweenness Centrality C_B(v) using Brandes' algorithm.
    NetworkX interprets 'weight' as distance (edge length along shortest paths).
    
    Parameters
    ----------
    G : nx.Graph
        Weighted undirected NetworkX graph.
    weight_attr : str
        Edge attribute to use as shortest-path distance (default: 'distance').
    normalized : bool
        Whether to normalize centrality values by 2 / ((n - 1) * (n - 2)).
    k_pivots : Optional[int]
        If specified, number of pivot nodes to sample for randomized Brandes approximation.
        If None, computes exact full Brandes betweenness centrality.
    seed : Optional[int]
        Random seed for pivot selection reproducibility.
        
    Returns
    -------
    Dict[str, float] : Node centrality dictionary.
    """
    return nx.betweenness_centrality(
        G,
        k=k_pivots,
        normalized=normalized,
        weight=weight_attr,
        seed=seed
    )


def compute_routing_redistribution(
    G_baseline: nx.Graph,
    G_stress: nx.Graph,
    k_pivots: Optional[int] = None,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes weighted Betweenness Centrality on identical topologies and derives
    the three core redistribution metrics for every gene v in the interactome:
    - Traffic Loss: ΔC_B+(v) = C_B,baseline(v) - C_B,stress(v)
    - Traffic Gain: ΔC_B-(v) = C_B,stress(v) - C_B,baseline(v)
    - Absolute Redistribution: |ΔC_B(v)| = |C_B,baseline(v) - C_B,stress(v)|
    
    Parameters
    ----------
    G_baseline : nx.Graph
        Scaffold graph with edge distance d_baseline.
    G_stress : nx.Graph
        Stress graph with edge distance d_stress on identical topology.
    k_pivots : Optional[int]
        Number of pivot samples for Brandes approximation, or None for exact Brandes.
    seed : Optional[int]
        Random seed for paired pivot sampling across both graphs.
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        df_redistribution : DataFrame with all genes, centrality scores, and redistribution metrics.
        stability_summary : Dictionary of sensitivity/stability metrics.
    """
    logger.info("Computing weighted Betweenness Centrality on G_baseline...")
    cb_baseline = compute_weighted_betweenness_centrality(
        G_baseline, weight_attr="distance", normalized=True, k_pivots=k_pivots, seed=seed
    )

    logger.info("Computing weighted Betweenness Centrality on G_stress (identical paired pivots)...")
    cb_stress = compute_weighted_betweenness_centrality(
        G_stress, weight_attr="distance", normalized=True, k_pivots=k_pivots, seed=seed
    )

    nodes = sorted(list(G_baseline.nodes()))
    records = []

    for v in nodes:
        c_base = float(cb_baseline.get(v, 0.0))
        c_str = float(cb_stress.get(v, 0.0))
        
        # Core ChassisNet redistribution metrics
        delta_plus = c_base - c_str               # Traffic Loss
        delta_minus = c_str - c_base              # Traffic Gain
        abs_delta = abs(c_base - c_str)           # Absolute Redistribution
        degree = int(G_baseline.degree[v])

        records.append({
            "gene_orf": v,
            "degree": degree,
            "cb_baseline": c_base,
            "cb_stress": c_str,
            "delta_cb_plus": delta_plus,
            "delta_cb_minus": delta_minus,
            "abs_delta_cb": abs_delta
        })

    df = pd.DataFrame(records)
    # Sort by absolute redistribution descending
    df = df.sort_values(by="abs_delta_cb", ascending=False).reset_index(drop=True)
    df["rank_abs_delta_cb"] = np.arange(1, len(df) + 1)
    df["rank_delta_cb_plus"] = df["delta_cb_plus"].rank(ascending=False, method="min").astype(int)
    df["rank_delta_cb_minus"] = df["delta_cb_minus"].rank(ascending=False, method="min").astype(int)

    logger.info(
        f"Redistribution calculated for {len(df):,} genes. "
        f"Top |ΔC_B| range: [{df['abs_delta_cb'].max():.6f}, {df['abs_delta_cb'].iloc[min(20, len(df)-1)]:.6f}]"
    )

    # Sensitivity check: PageRank redistribution
    stability_metrics = run_sensitivity_stability_check(G_baseline, G_stress, df)
    return df, stability_metrics


def run_sensitivity_stability_check(
    G_baseline: nx.Graph,
    G_stress: nx.Graph,
    df_redistribution: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluates rank stability of Betweenness Centrality redistribution against
    diffusion-based PageRank redistribution (|ΔPR|).
    """
    logger.info("Performing sensitivity check: evaluating rank stability against PageRank redistribution...")
    try:
        pr_baseline = nx.pagerank(G_baseline, weight="weight", max_iter=200, tol=1e-6)
        pr_stress = nx.pagerank(G_stress, weight="weight", max_iter=200, tol=1e-6)

        nodes = df_redistribution["gene_orf"].values
        abs_delta_cb = df_redistribution["abs_delta_cb"].values
        abs_delta_pr = np.array([abs(pr_baseline.get(v, 0.0) - pr_stress.get(v, 0.0)) for v in nodes])

        # Rank correlation between |ΔC_B| and |ΔPR|
        spearman_corr, spearman_p = stats.spearmanr(abs_delta_cb, abs_delta_pr)
        kendall_tau, kendall_p = stats.kendalltau(abs_delta_cb, abs_delta_pr)

        logger.info(
            f"Stability assessment: Spearman rho(|ΔC_B|, |ΔPR|) = {spearman_corr:.4f} "
            f"(p = {spearman_p:.2e}), Kendall tau = {kendall_tau:.4f}"
        )

        return {
            "spearman_rho_pagerank": float(spearman_corr),
            "spearman_pval_pagerank": float(spearman_p),
            "kendall_tau_pagerank": float(kendall_tau),
            "kendall_pval_pagerank": float(kendall_p),
            "method": "PageRank diffusion redistribution"
        }
    except Exception as e:
        logger.warning(f"PageRank sensitivity check encountered an exception: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    from data_ingestion import fetch_biogrid_physical, fetch_string_physical, fetch_transcriptomic_matrix
    from scaffold import build_physical_scaffold
    from reweighting import reweight_physical_edges, build_identical_topology_graphs

    logging.basicConfig(level=logging.INFO)
    df_bg = fetch_biogrid_physical()
    df_str = fetch_string_physical()
    G_base, scaffold_df = build_physical_scaffold(df_bg, df_str)
    expr_df, _ = fetch_transcriptomic_matrix()

    reweighted_df = reweight_physical_edges(scaffold_df, expr_df)
    G_b, G_s = build_identical_topology_graphs(reweighted_df)

    # Test with k=100 pivots for fast verification
    df_redist, stab = compute_routing_redistribution(G_b, G_s, k_pivots=100)
    print("Top 5 genes by |Delta_C_B|:")
    print(df_redist[["gene_orf", "degree", "cb_baseline", "cb_stress", "abs_delta_cb"]].head())
    print("Stability metrics:", stab)
