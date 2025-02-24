from collections.abc import Sequence
from typing import Any, Literal

import anndata as ad
import numpy as np
import sklearn.preprocessing as preprocessing
from numpy.typing import ArrayLike

from cfp._logging import logger
from cfp.data._utils import _to_list

__all__ = ["encode_onehot", "annotate_compounds", "get_molecular_fingerprints"]


def annotate_compounds(
    adata: ad.AnnData,
    compound_keys: str | Sequence[str],
    query_id_type: Literal["name", "cid"] = "name",
    obs_key_prefixes: str | Sequence[str] | None = None,
    copy: bool = False,
) -> None | ad.AnnData:
    """Annotates compounds in `adata` using pertpy and PubChem.

    Parameters
    ----------
    adata: ad.AnnData
        An :class:`~anndata.AnnData` object.
    compound_keys: str
        Key(s) in `adata.obs` containing the compound identifiers.
    query_id_type: str
        Type of the compound identifiers. Either "name" or "cid".
    obs_key_prefixes: str
        Prefix for the keys in `adata.obs` to store the annotations. If `None`, uses `compound_keys` as prefixes.
    copy: bool
        Return a copy of `adata` instead of updating it in place.

    Returns
    -------
        If `copy` is :obj:`True`, returns a new :class:`~anndata.AnnData` object with the compound annotations stored in `adata.obs`. Otherwise, updates `adata` in place.

        Sets the following fields for each value in `compound_keys`:
        `.obs[f"{obs_key_prefix}_pubchem_name"]`: Name of the compound.
        `.obs[f"{obs_key_prefix}_pubchem_ID"]`: PubChem CID of the compound.
        `.obs[f"{obs_key_prefix}_smiles"]`: SMILES representation of the compound.
    """
    try:
        import pertpy as pt
    except ImportError as e:
        raise ImportError(
            "pertpy is not installed. To annotate compounds, please install it via `pip install pertpy`."
        ) from e

    adata = adata.copy() if copy else adata

    compound_keys = _to_list(compound_keys)
    obs_key_prefixes = (
        _to_list(obs_key_prefixes) if obs_key_prefixes is not None else compound_keys
    )

    if len(compound_keys) != len(obs_key_prefixes):
        raise ValueError(
            "The number of `compound_keys` must match the number of values in `obs_key_prefixes`."
        )

    # Annotate compounds in each query column
    not_found = set()
    c_meta = pt.metadata.Compound()
    for query_key, prefix in zip(compound_keys, obs_key_prefixes, strict=False):
        c_meta.annotate_compounds(
            adata,
            query_id=query_key,
            query_id_type=query_id_type,
            verbosity=0,
            copy=False,
        )

        na_values = (
            adata.obs[query_key][adata.obs["smiles"].isna()]
            .astype(str)
            .unique()
            .tolist()
        )
        not_found.update(na_values)

        # Drop columns with new annotations
        adata.obs.drop(
            columns=[
                f"{prefix}_pubchem_name",
                f"{prefix}_pubchem_ID",
                f"{prefix}_smiles",
            ],
            errors="ignore",
            inplace=True,
        )

        # Rename with index to not overwrite existing columns
        adata.obs.rename(
            columns={
                "pubchem_name": f"{prefix}_pubchem_name",
                "pubchem_ID": f"{prefix}_pubchem_ID",
                "smiles": f"{prefix}_smiles",
            },
            inplace=True,
        )

    if not_found:
        logger.warning(
            f"Could not find annotations for the following compounds: {', '.join(not_found)}"
        )

    if copy:
        return adata


def _get_fingerprint(
    smiles: str, radius: int = 4, n_bits: int = 1024
) -> ArrayLike | None:
    """Computes Morgan fingerprints for a given SMILES string."""
    try:
        import rdkit.Chem.rdFingerprintGenerator as rfg
        from rdkit import Chem
    except ImportError:
        raise ImportError(
            "rdkit is not installed. To compute fingerprints, please install it via `pip install rdkit`."
        ) from None

    mmol = Chem.MolFromSmiles(str(smiles), sanitize=True)

    # Check if molecule is valid, MolFromSmiles returns None if error occurs
    if mmol is None:
        return None

    mfpgen = rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.array(mfpgen.GetFingerprint(mmol))


def get_molecular_fingerprints(
    adata,
    compound_keys: str,
    smiles_keys: str | None = None,
    uns_key_added: str = "fingerprints",
    radius: int = 4,
    n_bits: int = 1024,
    copy: bool = False,
) -> None | ad.AnnData:
    """Computes Morgan fingerprints for compounds in `adata` and stores them in `adata.uns`.

    Parameters
    ----------
    adata: ad.AnnData
        An :class:`~anndata.AnnData` object.
    compound_keys: str
        Key(s) in `adata.obs` containing the compound identifiers.
    smiles_keys: str
        Key(s) in `adata.obs` containing the SMILES strings. If `None`, uses `f"{compound_key}_smiles"`.
    uns_key_added: str
        Key in `adata.uns` to store the fingerprints.
    radius: int
        Radius of the Morgan fingerprints.
    n_bits: int
        Number of bits in the fingerprint.
    copy: bool
        Return a copy of `adata` instead of updating it in place

    Returns
    -------
        Updates `adata.uns` with the computed fingerprints.

        Sets the following fields:
        `.uns[uns_key_added]`: Dictionary containing the fingerprints for each compound.
    """
    adata = adata.copy() if copy else adata

    compound_keys = _to_list(compound_keys)

    if smiles_keys is None:
        smiles_keys = [f"{key}_smiles" for key in compound_keys]

    smiles_keys = _to_list(smiles_keys)

    # Get dict with SMILES for each compound
    smiles_dict = {}
    for compound_key, smiles_key in zip(compound_keys, smiles_keys, strict=False):  # type: ignore[arg-type]
        if compound_key not in adata.obs:
            raise KeyError(f"Key {compound_key} not found in `adata.obs`.")

        if smiles_key not in adata.obs:
            raise KeyError(f"Key {smiles_key} not found in `adata.obs`.")

        smiles_dict.update(adata.obs.set_index(compound_key)[smiles_key].to_dict())

    # Compute fingerprints for each compound
    valid_fingerprints = {}
    not_found = []
    for comp, smiles in smiles_dict.items():
        comp_fp = _get_fingerprint(smiles, radius=radius, n_bits=n_bits)
        if comp_fp is not None:
            valid_fingerprints[comp] = comp_fp
        else:
            not_found.append(str(comp))

    if not_found:
        logger.warning(
            f"Could not compute fingerprints for the following compounds: {', '.join(not_found)}"
        )

    adata.uns[uns_key_added] = valid_fingerprints

    if copy:
        return adata


def encode_onehot(
    adata: ad.AnnData,
    covariate_keys: str | Sequence[str],
    uns_key_added: str,
    exclude_values: str | Sequence[Any] = None,
    copy: bool = False,
) -> None | ad.AnnData:
    """Encodes covariates `adata.obs` as one-hot vectors and stores them in `adata.uns`.

    Parameters
    ----------
    adata : ad.AnnData
        An :class:`~anndata.AnnData` object.
    covariate_keys : str | Sequence[str]
        Key(s) in `adata.obs` containing the covariate(s) to encode.
    uns_key_added : str
        Key in `adata.uns` to store the one-hot encodings.
    exclude_values : str | Sequence[Any]
        Value(s) to exclude from the one-hot encoding.
    copy : bool
        Return a copy of `adata` instead of updating it in place.

    Returns
    -------
        If `copy` is :obj:`True`, returns a new :class:`~anndata.AnnData` object with the one-hot encodings stored in `adata.uns`. Otherwise, updates `adata` in place.

        Sets the following fields:
        `.uns[uns_key_added]`: Dictionary containing the one-hot encodings for each covariate.
    """
    adata = adata.copy() if copy else adata

    covariate_keys = _to_list(covariate_keys)
    exclude_values = _to_list(exclude_values)

    # Get unique values from all columns
    all_values = np.unique(adata.obs[covariate_keys].values.flatten())
    values_encode = np.setdiff1d(all_values, exclude_values).reshape(-1, 1)
    encoder = preprocessing.OneHotEncoder(sparse_output=False)
    encodings = encoder.fit_transform(values_encode)

    # Store encodings in adata.uns
    adata.uns[uns_key_added] = {}
    for value, encoding in zip(values_encode, encodings, strict=False):
        adata.uns[uns_key_added][value[0]] = encoding

    if copy:
        return adata
