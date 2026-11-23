import networkx as nx
import numpy as np
import pandas as pd
from flaky import flaky

from dowhy.gcm import (
    AdditiveNoiseModel,
    EmpiricalDistribution,
    InvertibleStructuralCausalModel,
    config,
    fit,
)
from dowhy.gcm.anomaly_in_distribution import find_root_causes_in_distribution
from dowhy.gcm.ml import create_linear_regressor
from dowhy.gcm.whatif import counterfactual_samples, interventional_samples

config.disable_progress_bars()

_ORDER = ["X0", "X1", "X2", "X3"]
_EQUATIONS = {"X1": ("X0", 2.0), "X2": ("X1", -1.5), "X3": ("X2", 1.0)}


def _generate_normal(num_samples: int) -> pd.DataFrame:
    noise = {node: scale * np.random.randn(num_samples) for node, scale in zip(_ORDER, [1.0, 0.5, 0.5, 0.5])}
    values = {"X0": noise["X0"]}
    for node, (parent, coefficient) in _EQUATIONS.items():
        values[node] = coefficient * values[parent] + noise[node]
    return pd.DataFrame(values)


def _plant_root_cause(root: str, shift: float) -> pd.DataFrame:
    noise = {node: scale * np.random.randn() for node, scale in zip(_ORDER, [1.0, 0.5, 0.5, 0.5])}
    values = {"X0": noise["X0"]}
    for node, (parent, coefficient) in _EQUATIONS.items():
        values[node] = coefficient * values[parent] + noise[node]
    values[root] += shift
    for node, (parent, coefficient) in _EQUATIONS.items():
        if _ORDER.index(node) > _ORDER.index(root):
            values[node] = coefficient * values[parent] + noise[node]
    return pd.DataFrame([values])


def _fitted_scm(normal_samples: pd.DataFrame) -> InvertibleStructuralCausalModel:
    scm = InvertibleStructuralCausalModel(nx.DiGraph([("X0", "X1"), ("X1", "X2"), ("X2", "X3")]))
    scm.set_causal_mechanism("X0", EmpiricalDistribution())
    for node in ["X1", "X2", "X3"]:
        scm.set_causal_mechanism(node, AdditiveNoiseModel(create_linear_regressor()))
    fit(scm, normal_samples)
    return scm


def test_identifies_anomaly_frontier_as_root_cause():
    np.random.seed(0)
    normal = _generate_normal(2000)
    scm = _fitted_scm(normal)

    # Inject the anomaly at X1; X2 and X3 become anomalous too, but downstream of an anomalous parent.
    anomaly = _plant_root_cause("X1", shift=8.0)
    scores = find_root_causes_in_distribution(scm, "X3", anomaly, normal)

    # Only X1 is on the anomaly frontier (anomalous node with non-anomalous parents).
    assert set(scores.keys()) == {"X1"}
    assert max(scores, key=scores.get) == "X1"
    assert scores["X1"] > 0


def test_interventional_estimate_differs_from_counterfactual():
    np.random.seed(0)
    normal = _generate_normal(2000)
    scm = _fitted_scm(normal)
    anomaly = _plant_root_cause("X1", shift=8.0)

    fix = {"X1": lambda _, value=float(normal["X1"].mean()): value}
    interventional_target = interventional_samples(causal_model=scm, interventions=fix, observed_data=anomaly)[
        "X3"
    ].to_numpy()[0]
    counterfactual_target = counterfactual_samples(causal_model=scm, interventions=fix, observed_data=anomaly)[
        "X3"
    ].to_numpy()[0]

    # The two estimators genuinely differ: IDI re-samples descendants in-distribution, the counterfactual
    # reuses the abducted (out-of-distribution) noise.
    assert abs(interventional_target - counterfactual_target) > 1e-6


def test_returns_empty_when_nothing_is_anomalous():
    np.random.seed(0)
    normal = _generate_normal(2000)
    scm = _fitted_scm(normal)

    # A fresh in-distribution sample has no root cause to report.
    normal_point = _generate_normal(1).iloc[[0]]
    assert find_root_causes_in_distribution(scm, "X3", normal_point, normal) == {}


@flaky(max_runs=3)
def test_recovers_planted_root_cause_across_trials():
    np.random.seed(2)
    normal = _generate_normal(2500)
    scm = _fitted_scm(normal)

    trials = 15
    hits = 0
    for _ in range(trials):
        root = np.random.choice(["X0", "X1", "X2"])
        anomaly = _plant_root_cause(root, shift=np.random.uniform(6.0, 10.0))
        scores = find_root_causes_in_distribution(scm, "X3", anomaly, normal)
        if scores and max(scores, key=scores.get) == root:
            hits += 1

    assert hits / trials >= 0.75
