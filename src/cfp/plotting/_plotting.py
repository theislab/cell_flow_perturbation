import types
from typing import Any, Literal

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from adjustText import adjust_text

from cfp import _constants
from cfp.plotting._utils import (
    _compute_kernel_pca_from_df,
    _compute_pca_from_df,
    _compute_umap_from_df,
    _get_colors,
    get_plotting_vars,
)


def plot_condition_embedding(
    adata: ad.AnnData,
    embedding: Literal["raw_embedding", "UMAP", "PCA", "Kernel_PCA"],
    dimensions: tuple[int, int] = (0, 1),
    hue: str | None = None,
    key: str = _constants.CONDITION_EMBEDDING,
    labels: list[str] = None,
    col_dict: dict[str, str] | None = None,
    title: str | None = None,
    show_lines: bool = False,
    show_text: bool = False,
    return_fig: bool = True,
    embedding_kwargs: dict[str, Any] = types.MappingProxyType({}),
    **kwargs: Any,
) -> mpl.figure.Figure:
    """Plot embedding of the conditions.

    Parameters
    ----------
        adata
            :class:`anndata.AnnData` object from a CellFlow model.
        embedding
            Embedding to plot. Options are "raw_embedding", "UMAP", "PCA", "Kernel_PCA".
        dimensions
            Dimensions of the embedding to plot.
        hue
            Covariate to color by.
        key
            Key in `adata.uns` where the embedding is stored. #TODO: correct
        labels
            TODO
        col_dict
            TODO
        title
            Title of the plot.
        show_lines
            Whether to show lines connecting points.
        show_text
            Whether to show text labels.
        return_fig
            Whether to return the figure.
        embedding_kwargs
            Additional keyword arguments for the embedding method.
        kwargs
            Additional keyword arguments for plotting.

    Returns
    -------
        :obj:`None` or :class:`matplotlib.figure.Figure`, depending on `return_fig`.
    """
    df = get_plotting_vars(adata, key=key)
    if embedding == "raw_embedding":
        emb = df[list(dimensions)]
    elif embedding == "UMAP":
        emb = _compute_umap_from_df(df, **embedding_kwargs)
    elif embedding == "PCA":
        emb = _compute_pca_from_df(df)
    elif embedding == "Kernel_PCA":
        emb = _compute_kernel_pca_from_df(df)
    else:
        raise ValueError(f"Embedding {embedding} not supported.")

    circle_size = kwargs.pop("circle_size", 40)
    circe_transparency = kwargs.pop("circe_transparency", 1.0)
    line_transparency = kwargs.pop("line_transparency", 0.8)
    line_width = kwargs.pop("line_width", 1.0)
    fontsize = kwargs.pop("fontsize", 9)
    fig_width = kwargs.pop("fig_width", 4)
    fig_height = kwargs.pop("fig_height", 4)
    labels_name = kwargs.pop("labels_name", None)
    axis_equal = kwargs.pop("axis_equal", None)

    sns.set_style("white")

    if labels is not None:
        if labels_name is None:
            labels_name = "labels"
        emb[labels_name] = labels

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()

    sns.despine(left=False, bottom=False, right=True)

    if (col_dict is None) and labels is not None:
        col_dict = _get_colors(labels)

    if hue is not None and hue not in df.index.names:
        raise ValueError(
            f"{hue} not found in index names. Valid values for `hue` are {df.index.names}."
        )

    sns.scatterplot(
        data=emb,
        x=dimensions[0],
        y=dimensions[1],
        hue=hue,
        palette=col_dict,
        alpha=circe_transparency,
        edgecolor="none",
        s=circle_size,
        ax=ax,
    )

    if show_lines:
        for i in range(len(emb)):
            if col_dict is None:
                ax.plot(
                    [0, emb[i, 0]],
                    [0, emb[i, 1]],
                    alpha=line_transparency,
                    linewidth=line_width,
                    c=None,
                )
            else:
                ax.plot(
                    [0, emb[i, 0]],
                    [0, emb[i, 1]],
                    alpha=line_transparency,
                    linewidth=line_width,
                    c=col_dict[labels[i]],
                )

    if show_text and labels is not None:
        texts = []
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            idx_label = np.where(labels == label)[0]
            texts.append(
                ax.text(
                    np.mean(emb[idx_label, 0]),
                    np.mean(emb[idx_label, 1]),
                    label,
                    fontsize=fontsize,
                )
            )

        adjust_text(
            texts,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.1),  # noqa: C408
            ax=ax,
        )

    if axis_equal:
        ax.axis("equal")
        ax.axis("square")

    title = title if title else embedding
    ax.set_title(title, fontsize=fontsize, fontweight="bold")

    ax.set_xlabel(f"dim {dimensions[0]}", fontsize=fontsize)
    ax.set_ylabel(f"dim {dimensions[1]}", fontsize=fontsize)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)

    return fig if return_fig else None
