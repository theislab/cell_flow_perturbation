from typing import Any, Literal

import jax
import jax.numpy as jnp
from ott.geometry import costs, geometry, graph, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

ScaleCost_t = float | Literal["mean", "max_cost", "median"]


def match_linear(
    source_batch: jnp.ndarray,
    target_batch: jnp.ndarray,
    cost_fn: costs.CostFn | None = costs.SqEuclidean(),
    epsilon: float | None = 1.0,
    scale_cost: ScaleCost_t = "mean",
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    threshold: float | None = None,
    **kwargs: Any,
) -> jnp.ndarray:
    """Compute solution to a linear OT problem.

    Parameters
    ----------
    source_batch
        Source point cloud of shape ``[n, d]``.
    target_batch
        Target point cloud of shape ``[m, d]``.
    cost_fn
        Cost function to use for the linear OT problem.
    epsilon
        Regularization parameter.
    scale_cost
        Scaling of the cost matrix.
    tau_a
        Parameter in :math:`(0, 1]` that defines how unbalanced the problem is
        in the source distribution. If :math:`1`, the problem is balanced in the source distribution.
    tau_b
        Parameter in :math:`(0, 1]` that defines how unbalanced the problem is in the target
        distribution. If :math:`1`, the problem is balanced in the target distribution.
    threshold
        Convergence criterion for the Sinkhorn algorithm.
    kwargs
        Additional arguments for :class:`ott.solvers.linear.sinkhorn.Sinkhorn`.

    Returns
    -------
    Optimal transport matrix between ``'source_batch'`` and ``'target_batch'``.
    """
    if threshold is None:
        threshold = 1e-3 if (tau_a == 1.0 and tau_b == 1.0) else 1e-2
    geom = pointcloud.PointCloud(
        source_batch,
        target_batch,
        cost_fn=cost_fn,
        epsilon=epsilon,
        scale_cost=scale_cost,
    )
    problem = linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b)
    solver = sinkhorn.Sinkhorn(threshold=threshold, **kwargs)
    out = solver(problem)
    return out.matrix


def _get_nearest_neighbors(
    X: jnp.ndarray, Y: jnp.ndarray | None = None, k: int = 30
) -> tuple[jnp.ndarray, jnp.ndarray]:
    concat = X if Y is None else jnp.concatenate((X, Y), axis=0)
    pairwise_euclidean_distances = pointcloud.PointCloud(concat, concat).cost_matrix
    distances, indices = jax.lax.approx_min_k(
        pairwise_euclidean_distances, k=k, recall_target=0.95, aggregate_to_topk=True
    )
    connectivities = jnp.multiply(jnp.exp(-distances), (distances > 0))
    return connectivities / jnp.sum(connectivities), indices


def _create_cost_matrix_lin(
    X: jnp.array,
    Y: jnp.array,
    k_neighbors: int,
) -> jnp.array:
    distances, indices = _get_nearest_neighbors(X, Y, k_neighbors)
    a = jnp.zeros((len(X) + len(Y), len(X) + len(Y)))
    adj_matrix = a.at[
        jnp.repeat(jnp.arange(len(X) + len(Y)), repeats=k_neighbors).flatten(),
        indices.flatten(),
    ].set(distances.flatten())
    return graph.Graph.from_graph(
        adj_matrix,
        normalize=True,
    ).cost_matrix[: len(X), len(X) :]


def match_linear_geodesic(
    source_batch: jnp.ndarray,
    target_batch: jnp.ndarray,
    epsilon: float = 1e-3,
    scale_cost: ScaleCost_t = "mean",
    tau_a: float = 1.0,
    tau_b: float = 1.0,
    k_neighbors: int | None = None,
    threshold: float | None = None,
    **kwargs,
) -> jnp.ndarray:
    """Compute the OT coupling based on a geodesic distance between source and target batch.

    Parameters
    ----------
    source_batch
        Source point cloud of shape ``[n, d]``.
    target_batch
        Target point cloud of shape ``[m, d]``.
    epsilon
        Regularization parameter.
    scale_cost
        Scaling of the cost matrix.
    tau_a
        Parameter in :math:`(0, 1]` that defines how unbalanced the problem is
        in the source distribution. If :math:`1`, the problem is balanced in the source distribution.
    tau_b
        Parameter in :math:`(0, 1]` that defines how unbalanced the problem is in the target
        distribution. If :math:`1`, the problem is balanced in the target distribution.
    threshold
        Convergence criterion for the Sinkhorn algorithm.
    kwargs
        Additional arguments for :class:`ott.solvers.linear.Sinkhorn`.

    Returns
    -------
    Geodesic distance between the two point clouds.
    """
    if threshold is None:
        threshold = 1e-3 if (tau_a == 1.0 and tau_b == 1.0) else 1e-2
    k_neighbors = len(source_batch) + 1 if k_neighbors is None else k_neighbors
    cm = _create_cost_matrix_lin(source_batch, target_batch, k_neighbors, **kwargs)
    geom = geometry.Geometry(cost_matrix=cm, epsilon=epsilon, scale_cost=scale_cost)
    solver = sinkhorn.Sinkhorn(threshold=threshold, **kwargs)
    out = solver(linear_problem.LinearProblem(geom, tau_a=tau_a, tau_b=tau_b))
    return out.matrix
