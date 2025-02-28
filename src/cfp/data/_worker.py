import numpy as np
from typing import Any, Dict
import pandas as pd
from collections import defaultdict
from ._utils import _check_shape, _pad_to_max_length


def _process_split_combination_worker(split_combination: list[Any], worker_data: Dict[str, Any]) -> Dict[str, Any]:
    # Extract data from worker_data
    perturb_covar_df = worker_data["perturb_covar_df"]
    split_covariates = worker_data["split_covariates"]

    # Initialize result containers
    condition_data = {}
    perturbation_idx_to_covariates = {}
    perturbation_idx_to_id = {}

    # Filter data for this split combination
    filter_dict = dict(zip(split_covariates, split_combination, strict=False))
    pc_df = perturb_covar_df[(perturb_covar_df[list(filter_dict.keys())] == list(filter_dict.values())).all(axis=1)]

    # Process each target condition
    for i, tgt_cond in pc_df.iterrows():
        tgt_cond = tgt_cond[worker_data["perturb_covar_keys"]]

        # Store condition mappings
        perturbation_idx_to_covariates[i] = tgt_cond.values
        perturbation_idx_to_id[i] = i

        if worker_data["is_conditional"]:
            # Process embeddings (simplified version of _get_perturbation_covariates)
            embedding = _get_condition_embeddings(tgt_cond, worker_data)
            for pert_cov, emb in embedding.items():
                if pert_cov not in condition_data:
                    condition_data[pert_cov] = []
                condition_data[pert_cov].append(emb)

    return {
        "condition_data": condition_data,
        "perturbation_idx_to_covariates": perturbation_idx_to_covariates,
        "perturbation_idx_to_id": perturbation_idx_to_id,
    }


def _get_condition_embeddings(condition_data: pd.Series, worker_data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Worker version of DataManager._get_perturbation_covariates."""
    perturb_covar_emb = defaultdict(list)

    # Get primary group from worker_data
    primary_group = worker_data["primary_group"]

    # Process primary covariates
    if primary_group:
        primary_covars = worker_data["perturbation_covariates"][primary_group]
        for primary_cov in primary_covars:
            value = condition_data[primary_cov]

            # Handle categorical/numeric differently
            if worker_data["is_categorical"]:
                cov_name = value
            else:
                cov_name = primary_cov

            # Get representation
            if primary_group in worker_data["covariate_reps"]:
                rep_key = worker_data["covariate_reps"][primary_group]
                # Convert cov_name to string for dictionary lookup
                cov_name_str = str(cov_name)
                if cov_name_str not in worker_data["rep_dict"][rep_key]:
                    # Handle missing representation
                    arr = np.full((1, 1), worker_data["null_value"])
                else:
                    arr = np.asarray(worker_data["rep_dict"][rep_key][cov_name_str])
            else:
                # If no representation is provided, use the value directly
                # But make sure it's a numeric value first
                arr = np.asarray(float(value) if not worker_data["is_categorical"] else worker_data["null_value"])

            # Only call _check_shape if arr is already a numeric array
            if not np.issubdtype(arr.dtype, np.number):
                arr = np.full((1, 1), worker_data["null_value"])
            else:
                arr = _check_shape(arr)
            perturb_covar_emb[primary_group].append(arr)

    # Process linked covariates
    for primary_cov, linked_groups in worker_data["linked_perturb_covars"].items():
        for linked_group, linked_cov in linked_groups.items():
            if linked_cov is None:
                arr = np.full((1, 1), worker_data["null_value"])
            else:
                value = condition_data[linked_cov]
                if linked_group in worker_data["covariate_reps"]:
                    rep_key = worker_data["covariate_reps"][linked_group]
                    # Convert value to string for dictionary lookup
                    value_str = str(value)
                    if value_str not in worker_data["rep_dict"][rep_key]:
                        arr = np.full((1, 1), worker_data["null_value"])
                    else:
                        arr = np.asarray(worker_data["rep_dict"][rep_key][value_str])
                else:
                    # If no representation is provided, use the value directly
                    # But make sure it's a numeric value first
                    arr = np.asarray(float(value) if np.issubdtype(type(value), np.number) else worker_data["null_value"])

            # Only call _check_shape if arr is already a numeric array
            if not np.issubdtype(arr.dtype, np.number):
                arr = np.full((1, 1), worker_data["null_value"])
            else:
                arr = _check_shape(arr)
            perturb_covar_emb[linked_group].append(arr)

    # Process sample covariates
    for sample_cov in worker_data["sample_covariates"]:
        value = condition_data[sample_cov]
        if sample_cov in worker_data["covariate_reps"]:
            rep_key = worker_data["covariate_reps"][sample_cov]
            # Convert value to string for dictionary lookup
            value_str = str(value)
            if value_str not in worker_data["rep_dict"][rep_key]:
                arr = np.full((1, 1), worker_data["null_value"])
            else:
                arr = np.asarray(worker_data["rep_dict"][rep_key][value_str])
        else:
            # If no representation is provided, use the value directly
            # But make sure it's a numeric value first
            arr = np.asarray(float(value) if np.issubdtype(type(value), np.number) else worker_data["null_value"])

        # Only call _check_shape if arr is already a numeric array
        if not np.issubdtype(arr.dtype, np.number):
            arr = np.full((1, 1), worker_data["null_value"])
        else:
            arr = _check_shape(arr)
            
        perturb_covar_emb[sample_cov].append(arr)

    # Pad and combine
    final_emb = {}
    for group, embeddings in perturb_covar_emb.items():
        padded = _pad_to_max_length(
            np.concatenate(embeddings, axis=0), worker_data["max_combination_length"], worker_data["null_value"]
        )
        final_emb[group] = padded

    return final_emb