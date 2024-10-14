import types
from typing import Any, Literal

import jax.numpy as jnp
import numpy as np
import pandas as pd
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.nonparametric import KCCA
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import KernelPCA

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
        Additional arguments for :class:`ott.solvers.linear.Sinkhorn`.

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


def predict_with_cca(
    embeddings_seen: pd.Series,
    target_variable: pd.Series,
    embeddings_unseen: pd.Series,
    kernel_pca_mode: Literal["linear", "poly", "rbf", "sigmoid", "cosine"],
    kernel_pca_n_components: int = 10,
) -> pd.Series:
    """Predict target variable for unseen data using canonical correlation analysis (CCA).

    Parameters
    ----------
    embeddings_seen
        Embeddings of the seen data. Index corresponding to condition names, values to embeddings.
    target_variable
        Target variable of the seen data. Index corresponding to condition names, values to target variable.
    embeddings_unseen
        Embeddings of the unseen data. Index corresponding to condition names, values to embeddings which to
        predict the target variable.
    kernel_pca_mode
        Kernel for the KernelPCA.
    kernel_pca_n_components
        Number of components to keep in the KernelPCA.


    Returns
    -------
    Predicted target variable for the unseen data.
    """
    kpca = KernelPCA(n_components=kernel_pca_n_components, kernel=kernel_pca_mode)
    X = embeddings_seen.values
    y = target_variable.loc[embeddings_seen.index].values

    X_transformed = kpca.fit_transform(X)

    cca = CCA(n_components=1)  # TODO: possibly extend to multiple variables
    cca.fit(X_transformed, y)
    _, y_c = cca.transform(X_transformed, y)
    correct_orientation = np.corrcoef((y_c.squeeze()), y)[0, 1] > 0.0

    X_new = embeddings_unseen.values
    X_new_transformed = kpca.transform(X_new)
    y_pred = cca.predict(X_new_transformed)

    return pd.Series(
        index=embeddings_unseen.index,
        data=y_pred * (1.0 if correct_orientation else -1.0),
    )


def predict_with_kernel_cca(
    embeddings_seen: pd.DataFrame,
    target_variable: pd.Series,
    embeddings_unseen: pd.DataFrame,
    kernel: Literal["linear", "poly", "rbf", "sigmoid", "cosine"] = "poly",
    kernel_kwargs: Any = types.MappingProxyType({}),
    return_all_data: bool = False,
) -> pd.Series | pd.DataFrame:
    """Predict target variable for unseen data using canonical correlation analysis (CCA).

    Parameters
    ----------
    embeddings_seen
        Embeddings of the seen data. Index corresponding to condition names, values to embeddings.
    target_variable
        Target variable of the seen data. Index corresponding to condition names, values to target variable.
    embeddings_unseen
        Embeddings of the unseen data. Index corresponding to condition names, values to embeddings which to
        predict the target variable.
    kernel
        Kernel for Kernel CCA.
    kernel_kwargs
        Keyword arguments for Kernel
    return_all_data
        Also returns the latent values for the seen embeddings


    Returns
    -------
    Predicted target variable for the unseen data.
    """
    X = embeddings_seen.values
    X_mean = X.mean(axis=0)
    X -= X_mean
    y = (
        target_variable.loc[embeddings_seen.index]
        .values.reshape(-1, 1)
        .astype("float64")
    )
    y_mean = y.mean(axis=0)
    y -= y_mean

    kcca = KCCA(latent_dimensions=1, kernel=kernel, **kernel_kwargs)
    kcca.fit((X, y))
    _, y_c = kcca.transform((X, y))
    correct_orientation = np.corrcoef((y_c.squeeze()), y.squeeze())[0, 1] > 0.0

    X_new = embeddings_unseen.values
    X -= X_mean
    y_pred = kcca.transform((X_new, None))[0]

    projections_unseen = pd.Series(
        index=embeddings_unseen.index,
        data=y_pred.squeeze() * (1.0 if correct_orientation else -1.0),
    )

    if not return_all_data:
        return projections_unseen

    projections_seen = pd.Series(
        index=embeddings_seen.index,
        data=y_c.squeeze() * (1.0 if correct_orientation else -1.0),
    ).to_frame(name="latent_dim")
    projections_seen["mode"] = "seen"
    projections_unseen = projections_unseen.to_frame(name="latent_dim")
    projections_unseen["mode"] = "unseen"
    return pd.concat((projections_seen, projections_unseen))


c_values = [0.9, 0.99]
default_hyperparameters = {
    "linear": {"kernel": ["linear"], "c": [c_values, c_values]},
    "poly": {
        "kernel": ["poly"],
        "degree": [[1, 5, 10, 20, 50], [1, 2, 3]],
        "c": [c_values, c_values],
    },
    "rbf": {
        "kernel": ["rbf"],
        "gamma": [[1.0, 1e-1, 1e-2], [1.0, 1e-1, 1e-2]],
        "c": [c_values, c_values],
    },
    "sigmoid": {"kernel": ["sigmoid"], "c": [c_values, c_values]},
    "cosine": {"kernel": ["sigmoid"], "c": [c_values, c_values]},
}


def kernel_cca_hyper(
    embeddings_seen: pd.DataFrame,
    target_variable: pd.Series,
    kernel: Literal["linear", "poly", "rbf", "sigmoid", "cosine"],
    param_grid: dict[str, list[Any]] | None = None,
    k_folds_cv: int = 5,
) -> tuple[Any, pd.Series]: #TODO: fix return type
    """TODO"""
    if param_grid is None:
        param_grid = default_hyperparameters[kernel]
    X = embeddings_seen.values
    X_mean = X.mean(axis=0)
    X -= X_mean
    y = (
        target_variable.loc[embeddings_seen.index]
        .values.reshape(-1, 1)
        .astype("float64")
    )
    y_mean = y.mean(axis=0)
    y -= y_mean

    kernel_reg_grid = GridSearchCV(
        KCCA(latent_dimensions=1), param_grid=param_grid, cv=k_folds_cv
    ).fit((X, y))
    return kernel_reg_grid.best_estimator_, pd.DataFrame(kernel_reg_grid.cv_results_)
