import time
import pandas as pd
import numpy as np
import pytest
from contextlib import contextmanager


@contextmanager
def timer(description: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"{description}: {elapsed:.4f} seconds")


def create_large_dataset(base_adata, scale_factor=10):
    """Create a larger dataset by replicating the base dataset."""
    import anndata as ad

    # Replicate the base dataset
    new_adata = ad.AnnData(np.vstack([base_adata.X for _ in range(scale_factor)]), dtype=base_adata.X.dtype)

    # Replicate obs
    new_obs = pd.concat([base_adata.obs for _ in range(scale_factor)])
    new_obs.index = [f"cell_{i}" for i in range(len(new_obs))]
    new_adata.obs = new_obs

    # Copy uns
    new_adata.uns = base_adata.uns.copy()

    return new_adata


def test_scaling(adata_perturbation, sizes=[10,], n_workers_list=[ 8,]):
    """Test how performance scales with data size and number of workers."""
    from cfp.data._datamanager import DataManager

    results = []
    base_size = len(adata_perturbation)

    # Use only the covariates that exist in the test dataset
    perturbation_covariates = {
        "drug": ["drug1", "drug2"],
        "dosage": ["dosage_a", "dosage_b"]
    }

    for size in sizes:
        print(f"\nTesting with {base_size * size} cells (scale factor: {size})")

        # Create larger dataset
        test_adata = create_large_dataset(adata_perturbation, size)

        dm = DataManager(
            test_adata,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps={"drug": "drug"},
            sample_covariates=["cell_type"],
            sample_covariate_reps={"cell_type": "cell_type"},
        )

        # Test sequential
        with timer("Sequential"):
            _ = dm.get_train_data(test_adata, parallelize=False)

        # Test parallel with different worker counts
        for n_workers in n_workers_list:
            with timer(f"Parallel ({n_workers} workers)"):
                _ = dm.get_train_data(
                    test_adata, 
                    parallelize=True,
                    n_workers=n_workers
                )

        print("-" * 50)


if __name__ == "__main__":
    # Run the test with the test dataset
    from conftest import adata_perturbation

    test_scaling(adata_perturbation())
