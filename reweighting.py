"""
reweighting.py
==============
Implements the smooth sigmoidal edge reweighting engine for ChassisNet:
1. Temporal Association Metric:
   - Pearson correlation r_uv if T >= 8.
   - Spearman rank correlation ρ_uv if 5 <= T < 8.
2. Statistical Significance Testing:
   - Continuous Student's t-transformation (d.f. = T - 2) to obtain exact p-values
     without the discrete floor trap of small-N permutations.
   - Optional empirical permutation test with adaptive early stopping.
3. Benjamini-Hochberg FDR Adjustment:
   - Derives q-values q_uv across all tested physical edges.
4. Continuous Logistic Weight Assignment:
   - W_stress(u, v) = W_baseline(u, v) * (1.0 / (1.0 + exp(-β * ρ_uv))) * (1.0 - q_uv)
     with steepness parameter β = 2.0.
   - Dynamic shortest-path distance:
     d_stress(u, v) = 1.0 / max(W_stress(u, v), 1e-6)
   - Strictly topology-preserving: Identical node and edge set E_physical evaluated.
"""

import logging
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import networkx as nx

logger = logging.getLogger("ChassisNet.Reweighting")

EPSILON = 1e-6
BETA_DEFAULT = 2.0


def compute_spearman_pvalues_student_t(rho: np.ndarray, df: int) -> np.ndarray:
    """
    Computes two-tailed p-values for Spearman correlation using Student's t-transformation:
    t = rho * sqrt((df) / (1 - rho^2)) with df = T - 2.
    
    Avoids discrete permutation floor traps for small T while providing continuous,
    well-calibrated p-values for FDR correction.
    """
    rho_clipped = np.clip(rho, -0.9999999999, 0.9999999999)
    t_stat = rho_clipped * np.sqrt(df / (1.0 - rho_clipped ** 2))
    p_vals = 2.0 * stats.t.sf(np.abs(t_stat), df=df)
    return np.clip(p_vals, 0.0, 1.0)


def compute_pearson_pvalues_student_t(r: np.ndarray, df: int) -> np.ndarray:
    """
    Computes two-tailed p-values for Pearson correlation using Student's t-test:
    t = r * sqrt((df) / (1 - r^2)) with df = T - 2.
    """
    r_clipped = np.clip(r, -0.9999999999, 0.9999999999)
    t_stat = r_clipped * np.sqrt(df / (1.0 - r_clipped ** 2))
    p_vals = 2.0 * stats.t.sf(np.abs(t_stat), df=df)
    return np.clip(p_vals, 0.0, 1.0)


def run_empirical_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
    early_stop_thresh: float = 0.1,
    min_permutations: int = 50,
    seed: Optional[int] = 42
) -> float:
    """
    Permutation test of temporal index for Spearman rank correlation with adaptive early stopping.
    """
    if len(x) != len(y):
        raise ValueError("Series lengths must match.")
    
    rng = np.random.RandomState(seed)
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    
    # Observed Spearman rho
    obs_rho, _ = stats.spearmanr(rx, ry)
    if np.isnan(obs_rho):
        return 1.0
    
    abs_obs = abs(obs_rho)
    count = 0
    
    for i in range(1, n_permutations + 1):
        perm_ry = rng.permutation(ry)
        perm_rho, _ = stats.spearmanr(rx, perm_ry)
        if abs(perm_rho) >= abs_obs:
            count += 1
            
        # Adaptive early stopping: if p > early_stop_thresh after min_permutations, stop early
        if i >= min_permutations and (count / i) > early_stop_thresh:
            return float((count + 1) / (i + 1))
            
    return float((count + 1) / (n_permutations + 1))


def reweight_physical_edges(
    scaffold_edges_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    use_replicates: bool = False,
    beta: float = BETA_DEFAULT,
    eps: float = EPSILON
) -> pd.DataFrame:
    """
    Executes smooth sigmoidal edge reweighting for all physical edges.
    
    Parameters
    ----------
    scaffold_edges_df : pd.DataFrame
        Scaffold edges containing ['interactor_a', 'interactor_b', 'W_baseline', 'd_baseline'].
    expr_df : pd.DataFrame
        Log2 expression matrix indexed by systematic ORF name.
    use_replicates : bool
        If True, use all available sample columns (e.g. WT + EV series, N=10).
        If False, use only WT time course (T=5).
    beta : float
        Sigmoid steepness parameter (default = 2.0).
    eps : float
        Small positive epsilon floor to prevent division by zero in distance.
        
    Returns
    -------
    pd.DataFrame
        Updated edges DataFrame containing correlation, p-value, q-value,
        W_stress, and d_stress for every edge in the scaffold.
    """
    logger.info("Executing continuous sigmoidal edge reweighting engine...")

    # Filter columns based on replicate setting
    if not use_replicates:
        wt_cols = [c for c in expr_df.columns if "WT" in c]
        active_expr = expr_df[wt_cols] if wt_cols else expr_df.iloc[:, :5]
    else:
        active_expr = expr_df

    T = active_expr.shape[1]
    logger.info(f"Temporal resolution: T = {T} time points/samples ({active_expr.columns.tolist()})")

    # Determine association metric
    use_pearson = (T >= 8)
    metric_name = "Pearson" if use_pearson else "Spearman"
    logger.info(f"Selected association metric based on T={T}: {metric_name} correlation")

    # Map genes to integer indices for rapid vectorized matrix lookup
    gene_to_idx = {gene: idx for idx, gene in enumerate(active_expr.index)}
    expr_matrix = active_expr.values.astype(np.float64)

    # Compute gene variances across time
    gene_vars = np.var(expr_matrix, axis=1)
    
    # Pre-rank expression matrix once for vectorized Spearman calculation
    if not use_pearson:
        ranked_matrix = np.apply_along_axis(stats.rankdata, 1, expr_matrix)
        # Center the ranked data
        data_centered = ranked_matrix - ranked_matrix.mean(axis=1, keepdims=True)
    else:
        data_centered = expr_matrix - expr_matrix.mean(axis=1, keepdims=True)

    norms = np.sqrt(np.sum(data_centered ** 2, axis=1))

    edges_df = scaffold_edges_df.copy()
    m_edges = len(edges_df)
    
    u_genes = edges_df["interactor_a"].values
    v_genes = edges_df["interactor_b"].values
    w_baseline = edges_df["W_baseline"].values.astype(np.float64)

    # Boolean mask of edges where both genes are quantified with non-zero variance
    valid_mask = np.zeros(m_edges, dtype=bool)
    u_indices = np.full(m_edges, -1, dtype=int)
    v_indices = np.full(m_edges, -1, dtype=int)

    for i in range(m_edges):
        u, v = u_genes[i], v_genes[i]
        if u in gene_to_idx and v in gene_to_idx:
            iu = gene_to_idx[u]
            iv = gene_to_idx[v]
            if gene_vars[iu] > 1e-9 and gene_vars[iv] > 1e-9:
                valid_mask[i] = True
                u_indices[i] = iu
                v_indices[i] = iv

    num_valid = int(np.sum(valid_mask))
    logger.info(
        f"Dynamic variance pre-filtering: {num_valid:,} of {m_edges:,} edges "
        f"({num_valid / m_edges * 100:.1f}%) have both interactors quantified with dynamic variation."
    )

    # Vectorized correlation computation across valid edges
    correlations = np.zeros(m_edges, dtype=np.float64)
    p_values = np.ones(m_edges, dtype=np.float64)
    q_values = np.zeros(m_edges, dtype=np.float64)

    if num_valid > 0:
        valid_iu = u_indices[valid_mask]
        valid_iv = v_indices[valid_mask]

        u_vecs = data_centered[valid_iu]  # (num_valid, T)
        v_vecs = data_centered[valid_iv]  # (num_valid, T)
        u_norms = norms[valid_iu]
        v_norms = norms[valid_iv]

        # Dot product
        numerators = np.sum(u_vecs * v_vecs, axis=1)
        denominators = np.maximum(u_norms * v_norms, 1e-12)
        valid_corrs = np.clip(numerators / denominators, -1.0, 1.0)
        correlations[valid_mask] = valid_corrs

        # Significance testing: Student's t-distribution approximation with d.f. = T - 2
        deg_freedom = max(T - 2, 1)
        if use_pearson:
            valid_pvals = compute_pearson_pvalues_student_t(valid_corrs, df=deg_freedom)
        else:
            valid_pvals = compute_spearman_pvalues_student_t(valid_corrs, df=deg_freedom)

        p_values[valid_mask] = valid_pvals

        # Benjamini-Hochberg FDR correction across tested edges
        _, valid_qvals, _, _ = multipletests(valid_pvals, method="fdr_bh")
        q_values[valid_mask] = valid_qvals

    # Continuous Logistic Weight Assignment:
    # W_stress(u, v) = W_baseline(u, v) * [1 / (1 + exp(-beta * corr))] * (1 - q)
    # For valid edges:
    logistic_mod = 1.0 / (1.0 + np.exp(-beta * correlations))
    confidence_mod = 1.0 - q_values

    # Unmeasured edges retain neutral baseline weight modulation:
    # For edges not in valid_mask, maintain W_stress = W_baseline * 0.5 (neutral sigmoid at rho=0, q=0)
    w_stress = w_baseline * logistic_mod * confidence_mod
    # Ensure numerical safety with minimum floor epsilon
    w_stress = np.maximum(w_stress, eps)
    d_stress = 1.0 / w_stress

    edges_df["association_metric"] = metric_name
    edges_df["correlation"] = correlations
    edges_df["p_value"] = p_values
    edges_df["q_value"] = q_values
    edges_df["W_stress"] = w_stress
    edges_df["d_stress"] = d_stress

    logger.info(
        f"Edge reweighting completed: mean W_baseline = {w_baseline.mean():.3f}, "
        f"mean W_stress = {w_stress.mean():.3f}"
    )
    logger.info(
        f"Significant co-expression (FDR q < 0.10): "
        f"{np.sum(q_values[valid_mask] < 0.10):,} edges; "
        f"(FDR q < 0.05): {np.sum(q_values[valid_mask] < 0.05):,} edges."
    )
    return edges_df


def build_identical_topology_graphs(
    reweighted_edges_df: pd.DataFrame
) -> Tuple[nx.Graph, nx.Graph]:
    """
    Constructs two NetworkX weighted graphs with identical node and edge sets:
    - G_baseline with edge distance attribute d_baseline.
    - G_stress with edge distance attribute d_stress.
    
    Parameters
    ----------
    reweighted_edges_df : pd.DataFrame
        DataFrame from reweight_physical_edges.
        
    Returns
    -------
    Tuple[nx.Graph, nx.Graph] : (G_baseline, G_stress)
    """
    logger.info("Constructing identical-topology NetworkX graphs G_baseline and G_stress...")

    G_baseline = nx.Graph()
    G_stress = nx.Graph()

    for _, row in reweighted_edges_df.iterrows():
        u = str(row["interactor_a"])
        v = str(row["interactor_b"])
        
        w_base = float(row["W_baseline"])
        d_base = float(row["d_baseline"])
        w_str = float(row["W_stress"])
        d_str = float(row["d_stress"])

        G_baseline.add_edge(u, v, weight=w_base, distance=d_base)
        G_stress.add_edge(u, v, weight=w_str, distance=d_str)

    # Verify identical topology
    assert G_baseline.number_of_nodes() == G_stress.number_of_nodes(), "Node count mismatch between baseline and stress graphs."
    assert G_baseline.number_of_edges() == G_stress.number_of_edges(), "Edge count mismatch between baseline and stress graphs."
    assert set(G_baseline.nodes()) == set(G_stress.nodes()), "Node sets are not identical."
    assert set(G_baseline.edges()) == set(G_stress.edges()), "Edge sets are not identical."

    logger.info(
        f"Identical-topology graphs successfully created: "
        f"{G_baseline.number_of_nodes():,} nodes and {G_baseline.number_of_edges():,} edges."
    )
    return G_baseline, G_stress


if __name__ == "__main__":
    from data_ingestion import fetch_biogrid_physical, fetch_string_physical, fetch_transcriptomic_matrix
    from scaffold import build_physical_scaffold

    logging.basicConfig(level=logging.INFO)
    df_bg = fetch_biogrid_physical()
    df_str = fetch_string_physical()
    G_base, scaffold_df = build_physical_scaffold(df_bg, df_str)
    expr_df, _ = fetch_transcriptomic_matrix()

    reweighted_df = reweight_physical_edges(scaffold_df, expr_df, use_replicates=False)
    G_b, G_s = build_identical_topology_graphs(reweighted_df)
    print("Verification completed: graphs built with exact topology.")
