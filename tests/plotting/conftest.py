import anndata as ad
import numpy as np
import pandas as pd
import pytest

from cfp import _constants


@pytest.fixture
def adata_with_condition_embedding(adata_perturbation) -> ad.AnnData:
    rng = np.random.default_rng(0)
    obs_cols = [
        "cell_type",
        "dosage",
        "drug1",
        "drug2",
        "drug3",
        "dosage_a",
        "dosage_b",
        "dosage_c",
    ]
    conditions = adata_perturbation.obs.drop_duplicates(subset=obs_cols)
    conditions = conditions.set_index(obs_cols)
    embedding = rng.random((len(conditions), 70))
    df = pd.DataFrame(data=embedding, columns=list(range(70)), index=conditions.index)
    df.index.set_names(obs_cols, inplace=True)
    adata_perturbation.uns[_constants.CFP_KEY] = {}
    adata_perturbation.uns[_constants.CFP_KEY][_constants.CONDITION_EMBEDDING] = df
    return adata_perturbation
