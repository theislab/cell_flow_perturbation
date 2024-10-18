import jax
import numpy as np
from jax import numpy as jnp
from jax.typing import ArrayLike
from ott.geometry import costs, pointcloud
from ott.tools.sinkhorn_divergence import sinkhorn_divergence
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.metrics.pairwise import rbf_kernel
from cfp.external import NegativeBinomial

__all__ = ["compute_metrics", "compute_metrics_fast", "compute_mean_metrics"]


def compute_r_squared(x: ArrayLike, y: ArrayLike) -> float:
    """Compute the R squared between true (x) and predicted (y)"""
    return r2_score(np.mean(x, axis=0), np.mean(y, axis=0))


def compute_sinkhorn_div(x: ArrayLike, y: ArrayLike, epsilon: float = 1e-2) -> float:
    """Compute the Sinkhorn divergence between x and y."""
    return float(
        sinkhorn_divergence(
            pointcloud.PointCloud,
            x=x,
            y=y,
            cost_fn=costs.SqEuclidean(),
            epsilon=epsilon,
            scale_cost=1.0,
        ).divergence
    )


def compute_e_distance(x: ArrayLike, y: ArrayLike) -> float:
    """Compute the energy distance as in Peidli et al."""
    sigma_X = pairwise_distances(x, x, metric="sqeuclidean").mean()
    sigma_Y = pairwise_distances(y, y, metric="sqeuclidean").mean()
    delta = pairwise_distances(x, y, metric="sqeuclidean").mean()
    return 2 * delta - sigma_X - sigma_Y


def compute_metrics(x: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Compute different metrics for x (true) and y (predicted)."""
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)
    metrics["sinkhorn_div_1"] = compute_sinkhorn_div(x, y, epsilon=1.0)
    metrics["sinkhorn_div_10"] = compute_sinkhorn_div(x, y, epsilon=10.0)
    metrics["sinkhorn_div_100"] = compute_sinkhorn_div(x, y, epsilon=100.0)
    metrics["e_distance"] = compute_e_distance(x, y)
    metrics["mmd"] = compute_scalar_mmd(x, y)
    return metrics


def compute_negbin_rec_loss(
    counts: ArrayLike, params: tuple[ArrayLike, ArrayLike]
) -> float:
    """Compute the reconstruction loss for the negative binomial noise model"""
    mean, theta = params
    px = NegativeBinomial(mean=mean, inverse_dispersion=jnp.exp(theta))
    loss = -px.log_prob(counts).sum(1).mean()
    return loss


def compute_mean_metrics(metrics: dict[str, dict[str, float]], prefix: str = ""):
    """Compute the mean value of different metrics."""
    metric_names = list(list(metrics.values())[0].keys())
    metric_dict: dict[str, list[float]] = {
        prefix + met_name: [] for met_name in metric_names
    }
    for met in metric_names:
        stat = 0.0
        for vals in metrics.values():
            stat += vals[met]
        metric_dict[prefix + met] = stat / len(metrics)
    return metric_dict


@jax.jit
def rbf_kernel_fast(x: ArrayLike, y: ArrayLike, gamma: float):
    xx = (x**2).sum(1)
    yy = (y**2).sum(1)
    xy = x @ y.T
    sq_distances = xx[:, None] + yy - 2 * xy
    return jnp.exp(-gamma * sq_distances)


def maximum_mean_discrepancy(
    x: ArrayLike, y: ArrayLike, gamma: float = 1.0, exact: bool = False
) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two samples: x and y.

    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        exact: a bool

    Returns
    -------
        a scalar denoting the squared maximum mean discrepancy loss.
    """
    kernel = rbf_kernel if exact else rbf_kernel_fast
    xx = kernel(x, x, gamma)
    xy = kernel(x, y, gamma)
    yy = kernel(y, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_scalar_mmd(
    x: ArrayLike, y: ArrayLike, gammas: float | None = None
) -> float:
    """Compute MMD across different length scales"""
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    mmds = [maximum_mean_discrepancy(x, y, gamma=gamma) for gamma in gammas]  # type: ignore[union-attr]
    return np.nanmean(np.array(mmds))


def compute_metrics_fast(x: ArrayLike, y: ArrayLike) -> dict[str, float]:
    """Compute metrics which are fast to compute."""
    metrics = {}
    metrics["r_squared"] = compute_r_squared(x, y)
    metrics["e_distance"] = compute_e_distance(x, y)
    metrics["mmd_distance"] = compute_scalar_mmd(x, y)
    return metrics
