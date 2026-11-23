"""Root-cause diagnosis via in-distribution interventions (IDI).

This implements the root-cause identification method of "Robust Root Cause Diagnosis using
In-Distribution Interventions" (Nagalapatti et al., ICLR 2025, https://arxiv.org/abs/2505.00930),
reimplemented against DoWhy's own GCM primitives.

The idea: a node is a root cause of an anomaly at the target iff (i) the node itself is anomalous
while its parents are not (it is on the "anomaly frontier"), and (ii) *fixing* it to an in-distribution
value removes the anomaly at the target. The contribution of each frontier node is the Shapley value of a
characteristic function that measures how much the target's anomaly score drops when a coalition of nodes
is fixed.

The key robustness choice is how the "fix" is evaluated. ``attribute_anomalies`` and other
counterfactual root-cause methods abduct the noise from the (rare, out-of-distribution) anomalous
sample and reuse it, which is unreliable precisely because the sample is anomalous. IDI instead evaluates
the fix with an *interventional* estimate (:func:`~dowhy.gcm.whatif.interventional_samples`), which
re-samples the fixed node's descendants in-distribution. Set ``use_counterfactual=True`` to recover the
counterfactual variant for comparison.
"""

from typing import Any, Callable, Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd

from dowhy.gcm.anomaly_scorer import AnomalyScorer
from dowhy.gcm.anomaly_scorers import MeanDeviationScorer
from dowhy.gcm.causal_models import InvertibleStructuralCausalModel, validate_causal_dag
from dowhy.gcm.shapley import ShapleyConfig, estimate_shapley_values
from dowhy.gcm.whatif import counterfactual_samples, interventional_samples
from dowhy.graph import get_ordered_predecessors, is_root_node

_NUM_SAMPLES_FIX_VALUE = 500


def find_root_causes_in_distribution(
    causal_model: InvertibleStructuralCausalModel,
    target_node: Any,
    anomaly_samples: pd.DataFrame,
    normal_samples: pd.DataFrame,
    anomaly_scorer_factory: Callable[[], AnomalyScorer] = MeanDeviationScorer,
    anomaly_quantile: float = 0.95,
    use_counterfactual: bool = False,
    shapley_config: Optional[ShapleyConfig] = None,
) -> Dict[Any, float]:
    """Diagnoses the root cause(s) of an anomaly at ``target_node`` using in-distribution interventions.

    :param causal_model: A fitted invertible SCM. ``use_counterfactual=True`` requires it to be invertible
                         (for noise abduction); the default in-distribution path only needs the mechanisms.
    :param target_node: The node whose anomaly is being diagnosed.
    :param anomaly_samples: One or more anomalous observations. If several rows are given, the single most
                            anomalous one for ``target_node`` (the "trigger") is diagnosed.
    :param normal_samples: Normal-operation data used to fit the per-node anomaly scorers and to estimate
                           in-distribution fix values.
    :param anomaly_scorer_factory: Callable returning a fresh per-node :class:`AnomalyScorer`.
    :param anomaly_quantile: A node is considered anomalous when its score exceeds this quantile of its
                             scores on ``normal_samples``.
    :param use_counterfactual: If True, evaluate the fix with counterfactuals (noise abduction) instead of
                               in-distribution interventions. Provided for comparison; IDI (the default)
                               is the robust choice.
    :param shapley_config: Configuration for the Shapley value estimation.
    :return: A dictionary mapping each frontier node to its root-cause score (higher = more responsible).
             Nodes that do not reduce the target anomaly are omitted.
    """
    validate_causal_dag(causal_model.graph)
    graph = causal_model.graph

    if target_node not in graph.nodes:
        raise ValueError(f"Target node {target_node!r} is not in the causal graph.")
    if anomaly_samples.shape[0] == 0:
        raise ValueError("anomaly_samples must contain at least one row.")

    scorers = _fit_anomaly_scorers(normal_samples, graph.nodes, anomaly_scorer_factory)
    thresholds = {
        node: float(np.quantile(scorers[node].score(normal_samples[node].to_numpy()), anomaly_quantile))
        for node in graph.nodes
    }

    # Pick the single most anomalous observation for the target (the "trigger" sample).
    target_scores_all = scorers[target_node].score(anomaly_samples[target_node].to_numpy())
    trigger = anomaly_samples.iloc[[int(np.argmax(target_scores_all))]]

    node_scores = {node: float(scorers[node].score(trigger[node].to_numpy())[0]) for node in graph.nodes}
    target_anomaly_score = node_scores[target_node]

    candidates = _anomaly_frontier(graph, target_node, node_scores, thresholds)
    if not candidates:
        return {}

    fix_values = {node: _in_distribution_fix_value(causal_model, node, trigger, normal_samples) for node in candidates}

    sampler = counterfactual_samples if use_counterfactual else interventional_samples

    def set_function(coalition: np.ndarray) -> float:
        interventions: Dict[Any, Callable[[np.ndarray], float]] = {}
        for index, is_selected in enumerate(coalition):
            if is_selected == 1:
                node = candidates[index]
                interventions[node] = _constant_intervention(fix_values[node])
        if not interventions:
            return 0.0
        fixed = sampler(causal_model=causal_model, interventions=interventions, observed_data=trigger)
        fixed_score = float(scorers[target_node].score(fixed[target_node].to_numpy())[0])
        return target_anomaly_score - fixed_score

    shapley_values = np.squeeze(estimate_shapley_values(set_function, len(candidates), shapley_config)).reshape(-1)

    return {candidates[i]: float(score) for i, score in enumerate(shapley_values) if score > 0}


def _fit_anomaly_scorers(
    normal_samples: pd.DataFrame, nodes, anomaly_scorer_factory: Callable[[], AnomalyScorer]
) -> Dict[Any, AnomalyScorer]:
    scorers = {}
    for node in nodes:
        scorer = anomaly_scorer_factory()
        scorer.fit(normal_samples[node].to_numpy())
        scorers[node] = scorer
    return scorers


def _anomaly_frontier(
    graph: nx.DiGraph, target_node: Any, node_scores: Dict[Any, float], thresholds: Dict[Any, float]
) -> list:
    """Ancestors of the target that are anomalous but have no anomalous parent."""
    frontier = []
    for node in nx.ancestors(graph, target_node):
        if node_scores[node] <= thresholds[node]:
            continue
        parents = get_ordered_predecessors(graph, node)
        if any(node_scores[parent] > thresholds[parent] for parent in parents):
            continue
        frontier.append(node)
    return frontier


def _in_distribution_fix_value(
    causal_model: InvertibleStructuralCausalModel, node: Any, trigger: pd.DataFrame, normal_samples: pd.DataFrame
) -> float:
    """The value the node would take in-distribution given the (abnormal) values of its parents."""
    if is_root_node(causal_model.graph, node):
        return float(normal_samples[node].mean())

    parents = get_ordered_predecessors(causal_model.graph, node)
    parent_values = np.tile(trigger[parents].to_numpy(), (_NUM_SAMPLES_FIX_VALUE, 1))
    drawn = causal_model.causal_mechanism(node).draw_samples(parent_values)
    return float(np.mean(drawn))


def _constant_intervention(value: float) -> Callable[[np.ndarray], float]:
    # Bind ``value`` now to avoid the late-binding closure trap inside the coalition loop.
    return lambda _, fixed=value: fixed
