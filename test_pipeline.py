"""
test_pipeline.py
================
Comprehensive automated unit and integration test suite for ChassisNet.
Validates:
1. Scaffold weight calculations and distance inversion.
2. Reweighting mathematics, significance testing, and FDR correction.
3. Strict graph topology preservation.
4. Metric consistency:
   np.isclose(abs_delta_cb, np.abs(delta_cb_plus))
   np.isclose(delta_cb_plus, -delta_cb_minus)
5. Directional hypergeometric enrichment calculations.
6. Standardized multivariable logistic regression.
7. Matched-control sampling and null model distributions.
8. End-to-end mini pipeline execution.
"""

import os
import unittest
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

from scaffold import compute_baseline_weight, build_physical_scaffold
from reweighting import (
    compute_spearman_pvalues_student_t,
    compute_pearson_pvalues_student_t,
    reweight_physical_edges,
    build_identical_topology_graphs
)
from graph_analytics import compute_routing_redistribution
from benchmarking import (
    run_hypergeometric_enrichment,
    evaluate_directional_overrepresentation,
    fit_multivariable_logistic_regression,
    run_matched_control_analysis
)


class TestChassisNetScaffold(unittest.TestCase):
    """Unit tests for scaffold construction and baseline weighting."""

    def test_baseline_weight_formula(self):
        # Case 1: In BioGRID only
        w1 = compute_baseline_weight(in_biogrid=True, string_score=0.0)
        self.assertAlmostEqual(w1, 0.50 + 0.25 * 1.0 + 0.0, places=5)
        self.assertAlmostEqual(w1, 0.75, places=5)

        # Case 2: In STRING only with score 800
        w2 = compute_baseline_weight(in_biogrid=False, string_score=800.0)
        self.assertAlmostEqual(w2, 0.50 + 0.0 + 0.25 * 0.8, places=5)
        self.assertAlmostEqual(w2, 0.70, places=5)

        # Case 3: In both with score 1000
        w3 = compute_baseline_weight(in_biogrid=True, string_score=1000.0)
        self.assertAlmostEqual(w3, 0.50 + 0.25 + 0.25, places=5)
        self.assertAlmostEqual(w3, 1.00, places=5)

        # Case 4: Neither (minimum theoretical baseline)
        w4 = compute_baseline_weight(in_biogrid=False, string_score=0.0)
        self.assertAlmostEqual(w4, 0.50, places=5)

    def test_scaffold_distance_inversion(self):
        df_bg = pd.DataFrame([{"interactor_a": "GENE_A", "interactor_b": "GENE_B"}])
        df_str = pd.DataFrame([{"interactor_a": "GENE_A", "interactor_b": "GENE_B", "string_score": 600}])
        G, edges = build_physical_scaffold(df_bg, df_str)

        w = edges.iloc[0]["W_baseline"]
        d = edges.iloc[0]["d_baseline"]
        self.assertAlmostEqual(d, 1.0 / w, places=6)
        self.assertAlmostEqual(w, 0.50 + 0.25 + 0.25 * 0.6, places=6)
        self.assertEqual(G.number_of_nodes(), 2)
        self.assertEqual(G.number_of_edges(), 1)


class TestChassisNetReweighting(unittest.TestCase):
    """Unit tests for edge reweighting and topology preservation."""

    def test_student_t_pvalues(self):
        # Perfect correlation -> p-value near 0
        rho_high = np.array([0.999, -0.999])
        p_high = compute_spearman_pvalues_student_t(rho_high, df=3)
        self.assertTrue((p_high < 0.01).all())

        # Zero correlation -> p-value near 1.0
        rho_zero = np.array([0.0])
        p_zero = compute_spearman_pvalues_student_t(rho_zero, df=3)
        self.assertAlmostEqual(p_zero[0], 1.0, places=3)

    def test_topology_preservation(self):
        # Create synthetic triangle scaffold
        edges_data = [
            {"interactor_a": "G1", "interactor_b": "G2", "in_biogrid": 1, "string_score": 500, "W_baseline": 0.875, "d_baseline": 1.0/0.875},
            {"interactor_a": "G2", "interactor_b": "G3", "in_biogrid": 0, "string_score": 700, "W_baseline": 0.675, "d_baseline": 1.0/0.675},
            {"interactor_a": "G1", "interactor_b": "G3", "in_biogrid": 1, "string_score": 0, "W_baseline": 0.75, "d_baseline": 1.0/0.75},
        ]
        scaffold_df = pd.DataFrame(edges_data)

        # Synthetic expression matrix (T = 5)
        expr_df = pd.DataFrame(
            data=[
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.1, 2.1, 2.9, 4.2, 4.8],
                [5.0, 4.0, 3.0, 2.0, 1.0],
            ],
            index=["G1", "G2", "G3"],
            columns=["WT_t0", "WT_t1", "WT_t2", "WT_t3", "WT_t4"]
        )

        reweighted_df = reweight_physical_edges(scaffold_df, expr_df, beta=2.0)
        G_b, G_s = build_identical_topology_graphs(reweighted_df)

        # Node and edge sets must be strictly identical
        self.assertEqual(set(G_b.nodes()), set(G_s.nodes()))
        self.assertEqual(set(G_b.edges()), set(G_s.edges()))
        self.assertEqual(G_b.number_of_edges(), 3)
        self.assertEqual(G_s.number_of_edges(), 3)

        # Distances must be positive and non-zero
        for _, _, d in G_s.edges(data=True):
            self.assertGreater(d["distance"], 0.0)
            self.assertGreater(d["weight"], 0.0)


class TestChassisNetGraphAnalytics(unittest.TestCase):
    """Unit tests for Brandes betweenness centrality and redistribution metrics."""

    def test_metric_consistency(self):
        """
        Validates mathematical consistency of redistribution metrics:
        assert np.isclose(abs_delta_cb, np.abs(delta_cb_plus)).all()
        assert np.isclose(delta_cb_plus, -delta_cb_minus).all()
        """
        G_baseline = nx.path_graph(5)
        G_stress = nx.path_graph(5)

        for u, v in G_baseline.edges():
            G_baseline[u][v]["distance"] = 1.0
            G_stress[u][v]["distance"] = 2.0

        df_redist, _ = compute_routing_redistribution(G_baseline, G_stress, k_pivots=None)

        abs_delta_cb = df_redist["abs_delta_cb"].values
        delta_cb_plus = df_redist["delta_cb_plus"].values
        delta_cb_minus = df_redist["delta_cb_minus"].values

        # Strict metric consistency assertions
        self.assertTrue(np.isclose(abs_delta_cb, np.abs(delta_cb_plus)).all())
        self.assertTrue(np.isclose(delta_cb_plus, -delta_cb_minus).all())
        self.assertTrue(np.isclose(abs_delta_cb, np.abs(delta_cb_minus)).all())


class TestChassisNetBenchmarking(unittest.TestCase):
    """Unit tests for validation suite: hypergeometric, Logit, and matched controls."""

    def test_hypergeometric_enrichment(self):
        # Known distribution: M=100, K=20, n=10, k=5
        # Expected successes = 2.0. With k=5, overrepresented
        p_val, fe, odds_ratio = run_hypergeometric_enrichment(
            k_success_in_sample=5,
            sample_size=10,
            K_success_in_population=20,
            population_size=100
        )
        self.assertAlmostEqual(fe, 5.0 / 2.0, places=4)
        self.assertGreater(odds_ratio, 1.0)
        self.assertLess(p_val, 0.05)

    def test_multivariable_logistic_regression_standardization(self):
        np.random.seed(42)
        N = 500
        # Highly disparate scales
        df = pd.DataFrame({
            "gene_orf": [f"ORF_{i}" for i in range(N)],
            "abs_delta_cb": np.random.exponential(0.001, size=N),    # Scale ~ 10^-3
            "abs_log2fc": np.random.normal(2.0, 1.0, size=N),         # Scale ~ 1
            "degree": np.random.geometric(0.02, size=N),              # Scale ~ 50
            "ethanol_phenotype": np.random.binomial(1, 0.3, size=N)
        })

        # Must fit cleanly without singular matrix error
        summary_df, model = fit_multivariable_logistic_regression(df)
        self.assertEqual(len(summary_df), 4)
        predictors = set(summary_df["predictor"])
        self.assertIn("|Delta_C_B| (Standardized)", predictors)
        self.assertIn("|log2FC| (Standardized)", predictors)
        self.assertIn("Degree (Standardized)", predictors)

        # Check odds ratio equals exp(coef)
        for _, row in summary_df.iterrows():
            self.assertAlmostEqual(row["odds_ratio"], np.exp(row["coef"]), places=4)
            self.assertLess(row["ci_95_lower"], row["odds_ratio"])
            self.assertGreater(row["ci_95_upper"], row["odds_ratio"])

    def test_matched_control_analysis(self):
        np.random.seed(42)
        N = 200
        df = pd.DataFrame({
            "gene_orf": [f"ORF_{i}" for i in range(N)],
            "abs_delta_cb": np.linspace(0.1, 0.001, N),
            "degree": np.random.randint(1, 50, size=N),
            "mean_expr": np.random.uniform(1, 10, size=N),
            "var_expr": np.random.uniform(0.1, 2, size=N),
            "ethanol_phenotype": np.random.binomial(1, 0.25, size=N)
        })

        mc = run_matched_control_analysis(df, top_pct=0.05, n_controls_per_candidate=20, seed=42)
        self.assertIn("candidate_pheno_rate", mc)
        self.assertIn("matched_control_mean_rate", mc)
        self.assertIn("enrichment_ratio", mc)
        self.assertIn("empirical_pval", mc)
        self.assertGreaterEqual(mc["empirical_pval"], 0.0)
        self.assertLessEqual(mc["empirical_pval"], 1.0)


class TestChassisNetIntegration(unittest.TestCase):
    """End-to-end integration test of ChassisNet pipeline on synthetic network."""

    def test_end_to_end_mini_pipeline(self):
        np.random.seed(42)
        # Create connected graph
        G = nx.cycle_graph(20)
        edges = []
        for u, v in G.edges():
            edges.append({
                "interactor_a": f"YAL{u:03d}W",
                "interactor_b": f"YAL{v:03d}W",
                "W_baseline": 0.8,
                "d_baseline": 1.0 / 0.8
            })
        scaffold_df = pd.DataFrame(edges)

        # Expression matrix (T=5)
        genes = [f"YAL{i:03d}W" for i in range(20)]
        expr_df = pd.DataFrame(
            np.random.normal(5, 1, size=(20, 5)),
            index=genes,
            columns=[f"WT_t{t}" for t in range(5)]
        )
        de_df = pd.DataFrame({
            "gene_orf": genes,
            "log2fc": np.random.normal(0, 1, size=20),
            "abs_log2fc": np.random.exponential(1, size=20)
        })
        pheno_df = pd.DataFrame({
            "gene_orf": genes[:5],
            "ethanol_phenotype": [1]*5,
            "phenotype_details": ["sensitive"]*5
        })

        # Reweight
        reweighted = reweight_physical_edges(scaffold_df, expr_df, beta=2.0)
        G_b, G_s = build_identical_topology_graphs(reweighted)

        # Redistribution
        df_redist, _ = compute_routing_redistribution(G_b, G_s, k_pivots=None)

        # Overrepresentation
        merged = df_redist.copy()
        merged["abs_log2fc"] = de_df.set_index("gene_orf")["abs_log2fc"]
        merged["degree"] = [G_b.degree[g] for g in merged["gene_orf"]]
        merged["ethanol_phenotype"] = [1 if g in set(pheno_df["gene_orf"]) else 0 for g in merged["gene_orf"]]
        
        enrich = evaluate_directional_overrepresentation(merged, top_pct=0.25)
        self.assertGreater(len(enrich), 0)
        self.assertIn("model_metric", enrich.columns)
        self.assertIn("hypergeom_pval", enrich.columns)


if __name__ == "__main__":
    unittest.main()
