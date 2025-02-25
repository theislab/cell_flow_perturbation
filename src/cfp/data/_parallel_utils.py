import numpy as np
import jax.numpy as jnp
import anndata
import pandas as pd
from typing import Any, Dict, List, Tuple, Union
from multiprocessing import Pool
from ._utils import _to_list

def process_single_condition(args):
    i, tgt_cond, dm, adata, covariate_data, perturb_covar_to_cells, control_mask, split_cov_mask, rep_dict = args
    result = {"tgt_cond": tgt_cond[dm._perturb_covar_keys], "mask": None, "skip": False, "embedding": None}

    # Process masks for train/validation
    if adata is not None:
        mask = covariate_data.index.isin(perturb_covar_to_cells[i])
        mask *= (1 - control_mask) * split_cov_mask
        mask = np.array(mask == 1)
        if mask.sum() == 0:
            result["skip"] = True
            return result
        result["mask"] = mask

    # Get embeddings if conditional
    if dm.is_conditional:
        embedding = dm._get_perturbation_covariates(
            condition_data=result["tgt_cond"],
            rep_dict=rep_dict,
            perturb_covariates={k: _to_list(v) for k, v in dm._perturbation_covariates.items()},
        )
        result["embedding"] = embedding

    return result


def parallel_process_conditions(
    pc_df: pd.DataFrame,
    dm: Any,
    adata: anndata.AnnData | None,
    covariate_data: pd.DataFrame,
    perturb_covar_to_cells: List,
    control_mask: np.ndarray,
    split_cov_mask: np.ndarray,
    rep_dict: Dict[str, Any],
    n_workers: int = 4
) -> List:
    # Prepare arguments for parallel processing
    process_args = [
        (i, tgt_cond, dm, adata, covariate_data, perturb_covar_to_cells, control_mask, split_cov_mask, rep_dict)
        for i, tgt_cond in pc_df.iterrows()
    ]

    # Process conditions in parallel
    pool = Pool(n_workers)
    try:
        results = pool.map(process_single_condition, process_args)
    finally:
        pool.close()
        pool.join()

    return results

def process_conditions(
    pc_df: pd.DataFrame,
    dm: Any,
    adata: anndata.AnnData | None,
    covariate_data: pd.DataFrame,
    perturb_covar_to_cells: List,
    control_mask: np.ndarray,
    split_cov_mask: np.ndarray,
    rep_dict: Dict[str, Any],
    perturbation_covariates_mask: np.ndarray,
    condition_data: Dict[str, List[np.ndarray]],
    tgt_counter: int = 0,
    n_workers: int = 4,
) -> Tuple[List, Dict, Dict, np.ndarray, Dict, int]:
    """
    Process conditions and aggregate results.

    Parameters
    ----------
    pc_df : pd.DataFrame
        DataFrame containing perturbation conditions
    dm : Any
        DataManager instance
    adata : anndata.AnnData | None
        AnnData object or None
    covariate_data : pd.DataFrame
        DataFrame containing covariate data
    perturb_covar_to_cells : List
        List mapping perturbation conditions to cells
    control_mask : np.ndarray
        Mask for control cells
    split_cov_mask : np.ndarray
        Mask for split covariates
    rep_dict : Dict[str, Any]
        Dictionary containing representations
    perturbation_covariates_mask : np.ndarray
        Mask for perturbation covariates
    condition_data : Dict[str, List[np.ndarray]]
        Dictionary to store condition data
    tgt_counter : int, optional
        Starting value for target counter, by default 0
    n_workers : int, optional
        Number of workers for parallel processing, by default 4

    Returns
    -------
    Tuple containing:
        - conditional_distributions
        - perturbation_idx_to_covariates
        - perturbation_idx_to_id
        - perturbation_covariates_mask
        - condition_data
        - tgt_counter
    """
    conditional_distributions = []
    perturbation_idx_to_covariates = {}
    perturbation_idx_to_id = {}

    # Process all conditions in parallel
    results = parallel_process_conditions(
        pc_df, dm, adata, covariate_data, perturb_covar_to_cells,
        control_mask, split_cov_mask, rep_dict, n_workers=n_workers
    )

    # Process results sequentially (this part needs to be sequential due to tgt_counter)
    for idx, result in enumerate(results):
        if result["skip"]:
            continue

        if result["mask"] is not None:
            perturbation_covariates_mask[result["mask"]] = tgt_counter

        conditional_distributions.append(tgt_counter)
        perturbation_idx_to_covariates[tgt_counter] = result["tgt_cond"].values

        if result["embedding"] is not None:
            for pert_cov, emb in result["embedding"].items():
                condition_data[pert_cov].append(emb)

        tgt_counter += 1

    return (
        conditional_distributions,
        perturbation_idx_to_covariates,
        perturbation_idx_to_id,
        perturbation_covariates_mask,
        condition_data,
        tgt_counter,
    )