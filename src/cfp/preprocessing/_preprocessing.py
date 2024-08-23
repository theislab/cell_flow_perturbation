from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import anndata as ad
import sklearn.preprocessing as preprocessing

from cfp._logging import logger

__all__ = ["encode_onehot"]


def encode_onehot(
    adata: ad.AnnData,
    covariate_keys: str | Sequence[str],
    uns_key: Sequence[str] = "onehot",
    exclude_values: str | Sequence[Any] = None,
    copy: bool = False,
) -> None | ad.AnnData:
    """Encodes covariates `adata.obs` as one-hot vectors and stores them in `adata.uns`.

    Args:
        adata: Annotated data matrix.
        covariate_keys: Keys of the covariates to encode.
        uns_key: Key in `adata.uns` to store the one-hot encodings.
        exclude_values: Values to exclude from encoding.
        copy: Return a copy of `adata` instead of updating it in place.

    Returns
    -------
        If `copy` is `True`, returns a new `AnnData` object with the one-hot encodings stored in `adata.uns`. Otherwise, updates `adata` in place.
    """
    adata = adata.copy() if copy else adata

    covariate_keys = _to_list(covariate_keys)
    exclude_values = _to_list(exclude_values)

    all_values = np.unique(adata.obs[covariate_keys].values.flatten())
    values_encode = np.setdiff1d(all_values, exclude_values).reshape(-1, 1)
    encoder = preprocessing.OneHotEncoder(sparse_output=False)
    encodings = encoder.fit_transform(values_encode)

    adata.uns[uns_key] = {}
    for value, encoding in zip(values_encode, encodings, strict=False):
        adata.uns[uns_key][value[0]] = encoding

    if copy:
        return adata


def pca(adata: ad.AnnData, n_comps: int = 50, copy: bool = False):
    """Performs PCA on the data matrix and stores the results in `adata`.

    Args:
        adata: Annotated data matrix.
        n_comps: Number of principal components to compute.
        copy: Return a copy of `adata` instead of updating it in place.

    Returns
    -------
        If `copy` is `True`, returns a new `AnnData` object with the PCA results stored in `adata.obsm`. Otherwise, updates `adata` in place.
    """
    adata = adata.copy() if copy else adata

    adata.varm["X_mean"] = adata.X.mean(axis=0).T
    adata.layers["X_centered"] = csr_matrix(adata.X.A - adata.varm["X_mean"].T)
    adata.obsm["X_pca"] = sc.pp.pca(
        adata.layers["X_centered"], zero_center=False, n_comps=n_comps
    )
    if copy:
        return adata


def get_fingerprint(smiles, radius: int = 4, n_bits: int = 1024):
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    try:
        Chem.SanitizeMol(m)
    except:
        return None
    mfpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    return mfpgen.GetFingerprint(m)


def get_fingerprints_from_dict(smiles, radius: int = 4, n_bits: int = 1024):
    fingerprints = {}
    for drug, smile in smiles.items():
        drug_fp = get_fingerprint(smile, radius=radius, n_bits=n_bits)

        if drug_fp is None:
            logger.warning(f"Could not generate fingerprint for {drug}")
            continue

        fingerprints[drug] = drug_fp

    return fingerprints


def get_smiles_from_name(drugs: Sequence[str]) -> pd.DataFrame:
    drugs = np.unique(drugs).tolist()
    smiles = {}
    not_found = []
    for drug in drugs:
        try:
            compounds = pcp.get_compounds(drug, "name")
        except:
            not_found.append(drug)
            continue
        else:
            if len(compounds) > 1:
                logger.info(f"Multiple compounds found for {drug}, taking first")
            if len(compounds) == 0:
                not_found.append(drug)
                continue
            smiles[drug] = compounds[0].canonical_smiles

    if len(not_found) > 0:
        logger.info(f"Could not find smiles for the following drugs: {not_found}")

    return smiles
