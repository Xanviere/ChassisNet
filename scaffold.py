"""
scaffold.py
===========
Constructs the continuous multi-tier replication-aware baseline physical interactome scaffold:
1. E_physical = E_BioGRID_physical ∪ E_STRING_experimental.
2. For each edge (u, v) ∈ E_physical:
   W_baseline(u, v) = 0.50 + 0.25 * I_BioGRID(u, v) + 0.25 * (S_STRING_exp(u, v) / 1000.0)
   where I_BioGRID ∈ {0, 1} and S_STRING_exp ∈ [400, 1000] (0 if absent).
3. Baseline shortest-path distance:
   d_baseline(u, v) = 1.0 / W_baseline(u, v)
"""

import os
import logging
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
import networkx as nx

logger = logging.getLogger("ChassisNet.Scaffold")


def compute_baseline_weight(in_biogrid: bool, string_score: float = 0.0) -> float:
    """
    Computes baseline physical weight W_baseline(u, v):
    W_baseline(u, v) = 0.50 + 0.25 * I_BioGRID(u, v) + 0.25 * (S_STRING_exp(u, v) / 1000.0)
    
    Parameters
    ----------
    in_biogrid : bool
        Indicator whether edge is present in BioGRID physical interactions.
    string_score : float
        STRING experimental score in [400, 1000], or 0.0 if absent from STRING.
        
    Returns
    -------
    float : W_baseline in [0.50, 1.00]
    """
    i_biogrid = 1.0 if in_biogrid else 0.0
    s_string = float(string_score) if string_score is not None and not np.isnan(string_score) else 0.0
    weight = 0.50 + 0.25 * i_biogrid + 0.25 * (s_string / 1000.0)
    return float(weight)


def build_physical_scaffold(
    df_biogrid: pd.DataFrame,
    df_string: pd.DataFrame
) -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Combines BioGRID and STRING into an unweighted/replicated baseline graph G_baseline = (V, E_physical).
    
    Parameters
    ----------
    df_biogrid : pd.DataFrame
        DataFrame with columns ['interactor_a', 'interactor_b'].
    df_string : pd.DataFrame
        DataFrame with columns ['interactor_a', 'interactor_b', 'string_score'].
        
    Returns
    -------
    Tuple[nx.Graph, pd.DataFrame]
        G_baseline : NetworkX Graph with edge attributes 'weight' (W_baseline) and 'distance' (d_baseline).
        edges_df   : DataFrame with all physical edges, weights, and distances.
    """
    logger.info("Constructing continuous multi-tier physical interactome scaffold...")

    # Ensure standardized ordering (u < v)
    bg_df = df_biogrid[["interactor_a", "interactor_b"]].copy()
    bg_df["in_biogrid"] = 1

    str_df = df_string[["interactor_a", "interactor_b", "string_score"]].copy()
    
    # Outer merge to form union E_physical = E_BioGRID ∪ E_STRING
    merged_edges = pd.merge(
        bg_df,
        str_df,
        on=["interactor_a", "interactor_b"],
        how="outer"
    )

    merged_edges["in_biogrid"] = merged_edges["in_biogrid"].fillna(0).astype(int)
    merged_edges["string_score"] = merged_edges["string_score"].fillna(0.0).astype(float)

    # Compute W_baseline and d_baseline
    w_baseline = (
        0.50
        + 0.25 * merged_edges["in_biogrid"].values
        + 0.25 * (merged_edges["string_score"].values / 1000.0)
    )
    d_baseline = 1.0 / w_baseline

    merged_edges["W_baseline"] = w_baseline
    merged_edges["d_baseline"] = d_baseline

    # Construct NetworkX Graph
    G_baseline = nx.Graph()
    for _, row in merged_edges.iterrows():
        u = str(row["interactor_a"])
        v = str(row["interactor_b"])
        w = float(row["W_baseline"])
        d = float(row["d_baseline"])
        G_baseline.add_edge(
            u, v,
            weight=w,
            distance=d,
            in_biogrid=int(row["in_biogrid"]),
            string_score=float(row["string_score"])
        )

    logger.info(
        f"Scaffold built: |V| = {G_baseline.number_of_nodes():,} nodes, "
        f"|E| = {G_baseline.number_of_edges():,} physical edges."
    )
    logger.info(
        f"Baseline weight summary: min={w_baseline.min():.3f}, "
        f"mean={w_baseline.mean():.3f}, max={w_baseline.max():.3f}"
    )
    return G_baseline, merged_edges


def get_scaffold_summary(G: nx.Graph) -> Dict[str, Any]:
    """Computes structural metrics for the physical interactome scaffold."""
    degrees = [d for _, d in G.degree()]
    weights = [data["weight"] for _, _, data in G.edges(data=True)]
    distances = [data["distance"] for _, _, data in G.edges(data=True)]
    n_components = nx.number_connected_components(G)
    largest_cc = max(nx.connected_components(G), key=len)

    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_connected_components": n_components,
        "largest_component_size": len(largest_cc),
        "mean_degree": float(np.mean(degrees)),
        "median_degree": float(np.median(degrees)),
        "max_degree": int(np.max(degrees)),
        "mean_W_baseline": float(np.mean(weights)),
        "std_W_baseline": float(np.std(weights)),
        "mean_d_baseline": float(np.mean(distances)),
    }


if __name__ == "__main__":
    from data_ingestion import fetch_biogrid_physical, fetch_string_physical
    logging.basicConfig(level=logging.INFO)
    df_bg = fetch_biogrid_physical()
    df_str = fetch_string_physical()
    G, edges = build_physical_scaffold(df_bg, df_str)
    summary = get_scaffold_summary(G)
    for k, v in summary.items():
        print(f"  {k}: {v}")
