import types
from collections.abc import Sequence
from typing import Any, Literal

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

from cfp import _constants, _logging
from cfp.plotting._utils import (
    _compute_kernel_pca_from_df,
    _compute_pca_from_df,
    _compute_umap_from_df,
    _get_colors,
    _grouped_df_to_standard,
    _is_numeric,
    _joyplot,
    _remove_na,
    get_plotting_vars,
)

__all__ = ["plot_condition_embedding", "plot_densities"]


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


def plot_densities(
    data: pd.DataFrame,
    features: Sequence[str],
    group_by: str | None = None,
    density_fit: Literal["log1p", "raw"] = "raw",
    ax: mpl.axes.Axes | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    xlabels: bool = False,
    ylabels: bool = True,
    xlabelsize: float | None = None,
    xrot: float | None = None,
    labels: Sequence[Any] = None,
    ylabelsize: float | None = None,
    yrot: float | None = None,
    hist: bool = False,
    bins: int = 10,
    fade: bool = False,
    ylim: Literal["max"] | tuple[float, float] | None = "max",
    fill: bool = True,
    linecolor: Any = None,
    overlap: float = 1.0,
    background: Any = None,
    range_style: Literal["all", "individual", "group"] | list[float] = "all",
    x_range: tuple[float, float] = None,
    title: str | None = None,
    colormap: str | mpl.colors.Colormap | None = None,
    color: Any = None,
    normalize: bool = True,
    grid: bool = False,
    return_fig: bool = True,
    **kwargs,
):
    """Plot kernel density estimations of expressions.

    This function is adapted from https://github.com/leotac/joypy/blob/master/joypy/joyplot.py

    Parameters
    ----------
    data
        :class:`pandas.DataFrame` object containing (predicted) expression values.
    features
        Features whose density to plot.
    group_by
        Column in ``'data'`` to group by.
    density_fit
        Type of density fit to use. If "raw", the kernel density estimation is plotted. If "log1p", the log1p
        transformed values of the densities are plotted.
    ax
        :class:`matplotlib.axes.Axes` used for plotting. If :obj:`None`, create a new one.
    figsize
        Size of the figure.
    dpi
        Dots per inch.
    xlabels
        Whether to show x-axis labels.
    ylabels
        Whether to show y-axis labels.
    xlabelsize
        Size of the x-axis labels.
    xrot
        Rotation (in degrees) of the x-axis labels.
    labels
        Sequence of labels for each density plot.
    ylabelsize
        Size of the y-axis labels.
    yrot
        Rotation (in degrees) of the y-axis labels.
    hist
        If :obj:`True`, plot a histogram, otherwise a density plot.
    bins
        Number of bins to use, only applicable if ``hist`` is :obj:`True`.
    fade
        If :obj:`True`, automatically sets different values of transparency of the density plots.
    ylim
        Limits of the y-axis.
    fill
        Whether to fill the density plots. If :obj:`False`, only the lines are plotted.
    linecolor: :mpltype:`color`
        Color of the contour lines.
    overlap
        Overlap between the density plots. The higher the value, the more overlap between densities.
    background: :mpltype:`color`
        Background color of the plot.
    range_style
        Style of the range. Options are

        - "all" - all density plots have the same range, autmoatically determined.
        - "individual" - every density plot has its own range, automatically determined.
        - "group" - each plot has a range that covers the whole group
        - type :obj:`list` - custom ranges for each density plot.

    x_range
        Custom range for the x-axis, shared across all density plots. If :obj:`None`, set via ``'range_style'``.
    title
        Title of the plot.
    colormap
        Colormap to use.
    color: :mpltype:`color`
        Color of the density plots.
    normalize
        Whether to normalize the densities.
    grid
        Whether to show the grid.
    return_fig
        Whether to return the figure.
    kwargs
        Additional keyword arguments for the plot.

    Returns
    -------
    :class:`matplotlib.figure.Figure` if ``'return_fig'`` is :obj:`True`, else :obj:`None`.
    """
    if group_by is not None and isinstance(data, pd.DataFrame):
        grouped = data.groupby(group_by)
        if features is None:
            features = list(data.columns)
            features.remove(group_by)
        converted, _labels, sublabels = _grouped_df_to_standard(grouped, features)  # type: ignore[arg-type]
        if labels is None:
            labels = _labels
    elif isinstance(data, pd.DataFrame):
        if features is not None:
            data = data[features]
        converted = [
            [_remove_na(data[col])] for col in data.columns if _is_numeric(data[col])
        ]
        labels = [col for col in data.columns if _is_numeric(data[col])]
        sublabels = None
    else:
        raise TypeError(f"Unknown type for 'data': {type(data)!r}")

    if ylabels is False:
        labels = None

    if all(len(subg) == 0 for g in converted for subg in g):
        raise ValueError(
            "No numeric values found. Joyplot requires at least a numeric column/group."
        )

    if any(len(subg) == 0 for g in converted for subg in g):
        _logging.logger.warning("At least a column/group has no numeric values.")

    fig, axes = _joyplot(
        converted,
        labels=labels,
        sublabels=sublabels,
        grid=grid,
        xlabelsize=xlabelsize,
        xrot=xrot,
        ylabelsize=ylabelsize,
        yrot=yrot,
        ax=ax,
        dpi=dpi,
        figsize=figsize,
        hist=hist,
        bins=bins,
        fade=fade,
        ylim=ylim,
        fill=fill,
        linecolor=linecolor,
        overlap=overlap,
        background=background,
        xlabels=xlabels,
        range_style=range_style,
        x_range=x_range,
        title=title,
        colormap=colormap,
        color=color,
        normalize=normalize,
        density_fit=density_fit,
        **kwargs,
    )
    return fig if return_fig else None
