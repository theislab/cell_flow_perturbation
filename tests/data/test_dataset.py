import anndata as ad
import jax
import pytest
import numpy as np


class TestPerturbationData:
    @pytest.mark.parametrize(
        "cell_data",
        ["X", "X_pca", {"obsm": "X_pca"}, {"layers": "my_counts"}],
    )
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("control_data", [("drug1", "control")])
    @pytest.mark.parametrize("obs_perturbation_covariates", [[], [("dosage",)]])
    @pytest.mark.parametrize("uns_perturbation_covariates", [{}, {"drug": ("drug1",)}])
    def test_load_from_adata_no_combinations(
        self,
        adata_perturbation: ad.AnnData,
        cell_data,
        split_covariates,
        control_data,
        obs_perturbation_covariates,
        uns_perturbation_covariates,
    ):
        from cfp.data.data import PerturbationData

        pdata = PerturbationData.load_from_adata(
            adata_perturbation,
            cell_data=cell_data,
            split_covariates=split_covariates,
            control_data=control_data,
            obs_perturbation_covariates=obs_perturbation_covariates,
            uns_perturbation_covariates=uns_perturbation_covariates,
        )
        assert isinstance(pdata, PerturbationData)
        if split_covariates == []:
            assert pdata.n_controls == 1
        if split_covariates == ["cell_type"]:
            assert pdata.n_controls == len(
                adata_perturbation.obs["cell_type"].cat.categories
            )
        if obs_perturbation_covariates == [] and uns_perturbation_covariates == {}:
            assert pdata.n_perturbations == pdata.n_controls
            assert pdata.condition_data is None
            assert pdata.max_combination_length == 0
        else:
            assert isinstance(pdata.condition_data, dict)
            assert isinstance(pdata.condition_data[0], jax.Array)
            assert pdata.max_combination_length == 1

        if (
            obs_perturbation_covariates == ["dosage"]
            and uns_perturbation_covariates == []
        ):
            assert pdata.n_perturbations == len(
                adata_perturbation.obs["dosage"].cat.categories
            )
        if obs_perturbation_covariates == [] and uns_perturbation_covariates == {
            "drug": ("drug1",)
        }:
            assert (
                pdata.n_perturbations
                == (len(adata_perturbation.obs["drug1"].cat.categories) - 1)
                * pdata.n_controls
            )
        assert isinstance(pdata.cell_data, jax.Array)
        assert isinstance(pdata.split_covariates_mask, jax.Array)
        assert isinstance(pdata.split_idx_to_covariates, dict)
        assert isinstance(pdata.perturbation_covariates_mask, jax.Array)
        assert isinstance(pdata.perturbation_idx_to_covariates, dict)
        for value in pdata.pert_embedding_idx_to_covariates.values():
            assert isinstance(value, tuple) or isinstance(value, str)
            if isinstance(value, tuple):
                for el in value:
                    assert isinstance(el, str)
        assert isinstance(pdata.control_to_perturbation, dict)

    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("control_data", [("drug1", "control")])
    @pytest.mark.parametrize("obs_perturbation_covariates", [[], [("dosage",)]])
    @pytest.mark.parametrize(
        "uns_perturbation_covariates", [{"drug": ("drug1", "drug2")}]
    )
    def test_load_from_adata_with_combinations(
        self,
        adata_perturbation: ad.AnnData,
        split_covariates,
        control_data,
        obs_perturbation_covariates,
        uns_perturbation_covariates,
    ):
        from cfp.data.data import PerturbationData

        pdata = PerturbationData.load_from_adata(
            adata_perturbation,
            cell_data="X",
            split_covariates=split_covariates,
            control_data=control_data,
            obs_perturbation_covariates=obs_perturbation_covariates,
            uns_perturbation_covariates=uns_perturbation_covariates,
        )
        assert isinstance(pdata, PerturbationData)
        if split_covariates == []:
            assert pdata.n_controls == 1
        if split_covariates == ["cell_type"]:
            assert pdata.n_controls == len(
                adata_perturbation.obs["cell_type"].cat.categories
            )
        if obs_perturbation_covariates == [] and uns_perturbation_covariates == {}:
            assert pdata.n_perturbations == pdata.n_controls
            assert pdata.condition_data is None
            assert pdata.max_combination_length == 0
        else:
            assert isinstance(pdata.condition_data, dict)
            assert isinstance(pdata.condition_data[0], jax.Array)

            assert pdata.max_combination_length == 2

        if (
            obs_perturbation_covariates == ["dosage"]
            and uns_perturbation_covariates == []
        ):
            assert pdata.n_perturbations == len(
                adata_perturbation.obs["dosage"].cat.categories
            )
        if obs_perturbation_covariates == [] and uns_perturbation_covariates == {
            "drug": ("drug1",)
        }:
            assert (
                pdata.n_perturbations
                == (len(adata_perturbation.obs["drug1"].cat.categories) - 1)
                * pdata.n_controls
            )
        assert isinstance(pdata.cell_data, jax.Array)
        assert isinstance(pdata.split_covariates_mask, jax.Array)
        assert isinstance(pdata.split_idx_to_covariates, dict)
        assert isinstance(pdata.perturbation_covariates_mask, jax.Array)
        assert isinstance(pdata.perturbation_idx_to_covariates, dict)
        for value in pdata.pert_embedding_idx_to_covariates.values():
            assert isinstance(value, tuple) or isinstance(value, str)
            if isinstance(value, tuple):
                for el in value:
                    assert isinstance(el, str)
        assert isinstance(pdata.control_to_perturbation, dict)

    @pytest.mark.parametrize("el_to_delete", ["drug1", "cell_line_a"])
    def raise_wrong_uns_dict(self, adata_perturbation: ad.AnnData, el_to_delete):
        from cfp.data.data import PerturbationData

        cell_data = "X"
        split_covariates = ["cell_type"]
        control_data = ("drug1", "control")
        obs_perturbation_covariates = [("dosage",)]
        uns_perturbation_covariates = {
            "drug": ("drug1", "drug2"),
            "cell_type": ("cell_type",),
        }
        if el_to_delete == "drug1":
            del adata_perturbation.uns["drug"]["drug1"]
        if el_to_delete == "cell_line_a":
            del adata_perturbation.uns["cell_type"]["cell_line_a"]

        with pytest.raises(KeyError):
            _ = PerturbationData.load_from_adata(
                adata_perturbation,
                cell_data=cell_data,
                split_covariates=split_covariates,
                control_data=control_data,
                obs_perturbation_covariates=obs_perturbation_covariates,
                uns_perturbation_covariates=uns_perturbation_covariates,
            )


    @pytest.mark.parametrize("null_token", [0.0, -99.0])
    def test_for_masks(self, adata_perturbation_with_nulls, null_token):
        from cfp.data.data import PerturbationData
        adata = adata_perturbation_with_nulls
        cell_data = "X"
        split_covariates = ["cell_type"]
        control_data = ("drug1", "control")
        obs_perturbation_covariates = [("dosage",)]
        uns_perturbation_covariates = {"drug": ("drug1",)}

        null_value = "no_drug"

        pdata = PerturbationData.load_from_adata(
            adata,
            cell_data=cell_data,
            split_covariates=split_covariates,
            control_data=control_data,
            obs_perturbation_covariates=obs_perturbation_covariates,
            uns_perturbation_covariates=uns_perturbation_covariates,
            null_value=null_value,
            null_token=null_token,
        )

        
        assert isinstance(pdata.pert_embedding_idx_to_covariates, dict)
        correct_dict = {
            0: ("dosage",),
            1: "drug",
        }
        assert pdata.pert_embedding_idx_to_covariates == correct_dict
        
        perturbation_to_ix_mask = {k: np.array([el == null_value for el in v]) for k,v in pdata.perturbation_idx_to_covariates.items()}
        
        for perturbation_idx, is_null_perturbation_covariates_mask in perturbation_to_ix_mask.items():
            for i, is_null_perturbation_covariate in enumerate(is_null_perturbation_covariates_mask):
                condition_data_is_masked = bool(np.all(pdata.condition_data[i][perturbation_idx] == null_token, axis=1))
                assert condition_data_is_masked == bool(is_null_perturbation_covariate)