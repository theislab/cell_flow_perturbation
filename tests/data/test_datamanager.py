import anndata as ad
import jax
import pytest
import numpy as np

from cfp.data._datamanager import DataManager

perturbation_covariates_args = [
    {"drug": ["drug1"]},
    {"drug": ["drug1"], "dosage": ["dosage_a"]},
    {
        "drug": ["drug_a"],
        "dosage": ["dosage_a"],
    },
]

perturbation_covariate_comb_args = [
    {"drug": ["drug1", "drug2"]},
    {"drug": ["drug1", "drug2"], "dosage": ["dosage_a", "dosage_b"]},
    {
        "drug": ["drug_a", "drug_b", "drug_c"],
        "dosage": ["dosage_a", "dosage_b", "dosage_c"],
    },
]


@pytest.fixture(
    params=[
        "mask",
    ]
)
def algorithm(request):
    return request.param


class TestDataManager:
    @pytest.mark.parametrize("sample_rep", ["X", "X_pca"])
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariates_args)
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    @pytest.mark.parametrize("sample_covariates", [[], ["dosage_c"]])
    def test_init_DataManager(
        self,
        adata_perturbation: ad.AnnData,
        sample_rep,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
        sample_covariates,
    ):
        from cfp.data._datamanager import DataManager

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
        )
        assert isinstance(dm, DataManager)
        assert dm._sample_rep == sample_rep
        assert dm._control_key == "control"
        assert dm._split_covariates == split_covariates
        assert dm._perturbation_covariates == perturbation_covariates
        assert dm._sample_covariates == sample_covariates

    @pytest.mark.parametrize("el_to_delete", ["drug", "cell_type"])
    def test_raise_false_uns_dict(self, adata_perturbation: ad.AnnData, el_to_delete):
        from cfp.data._datamanager import DataManager

        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = "control"
        perturbation_covariates = {"drug": ("drug_a", "drug_b")}
        perturbation_covariate_reps = {"drug": "drug"}
        sample_covariates = ["cell_type"]
        sample_covariate_reps = {"cell_type": "cell_type"}

        if el_to_delete == "drug":
            del adata_perturbation.uns["drug"]
        if el_to_delete == "cell_type":
            del adata_perturbation.uns["cell_type"]

        with pytest.raises(ValueError, match=r".*representation.*not found.*"):
            _ = DataManager(
                adata_perturbation,
                sample_rep=sample_rep,
                split_covariates=split_covariates,
                control_key=control_key,
                perturbation_covariates=perturbation_covariates,
                perturbation_covariate_reps=perturbation_covariate_reps,
                sample_covariates=sample_covariates,
                sample_covariate_reps=sample_covariate_reps,
            )

    @pytest.mark.parametrize("el_to_delete", ["drug_b", "dosage_a"])
    def test_raise_covar_mismatch(self, adata_perturbation: ad.AnnData, el_to_delete):
        from cfp.data._datamanager import DataManager

        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = "control"
        perturbation_covariate_reps = {"drug": "drug"}
        perturbation_covariates = {
            "drug": ["drug_a", "drug_b"],
            "dosage": ["dosage_a", "dosage_b"],
        }
        if el_to_delete == "drug_b":
            perturbation_covariates["drug"] = ["drug_b"]
        if el_to_delete == "dosage_a":
            perturbation_covariates["dosage"] = ["dosage_b"]

        with pytest.raises(ValueError, match=r".*perturbation covariate groups must match.*"):
            _ = DataManager(
                adata_perturbation,
                sample_rep=sample_rep,
                split_covariates=split_covariates,
                control_key=control_key,
                perturbation_covariates=perturbation_covariates,
                perturbation_covariate_reps=perturbation_covariate_reps,
            )

    def test_raise_target_without_source(self, adata_perturbation: ad.AnnData):
        from cfp.data._datamanager import DataManager

        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = "control"
        perturbation_covariate_reps = {"drug": "drug"}
        perturbation_covariates = {
            "drug": ["drug_a", "drug_b"],
            "dosage": ["dosage_a", "dosage_b"],
        }

        adata_perturbation.obs.loc[
            (~adata_perturbation.obs["control"]) & (adata_perturbation.obs["cell_type"] == "cell_line_a"),
            "cell_type",
        ] = "cell_line_b"

        with pytest.raises(
            ValueError,
            match=r"Source distribution with split covariate values \{\('cell_line_a',\)\} do not have a corresponding target distribution.",
        ):
            _ = DataManager(
                adata_perturbation,
                sample_rep=sample_rep,
                split_covariates=split_covariates,
                control_key=control_key,
                perturbation_covariates=perturbation_covariates,
                perturbation_covariate_reps=perturbation_covariate_reps,
            )

    @pytest.mark.parametrize("sample_rep", ["X", "X_pca"])
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariates_args)
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    @pytest.mark.parametrize("sample_covariates", [[], ["dosage_c"]])
    def test_get_train_data(
        self,
        adata_perturbation: ad.AnnData,
        sample_rep,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
        sample_covariates,
        algorithm,
    ):
        from cfp.data._data import TrainingData
        from cfp.data._datamanager import DataManager

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
        )
        assert isinstance(dm, DataManager)
        assert dm._sample_rep == sample_rep
        assert dm._control_key == "control"
        assert dm._split_covariates == split_covariates
        assert dm._perturbation_covariates == perturbation_covariates
        assert dm._sample_covariates == sample_covariates

        train_data = dm.get_train_data(adata_perturbation, algorithm=algorithm)
        assert isinstance(train_data, TrainingData)
        assert isinstance(train_data, TrainingData)
        assert ((train_data.perturbation_covariates_mask == -1) + (train_data.split_covariates_mask == -1)).all()
        if split_covariates == []:
            assert train_data.n_controls == 1
        if split_covariates == ["cell_type"]:
            assert train_data.n_controls == len(adata_perturbation.obs["cell_type"].cat.categories)

        assert isinstance(train_data.condition_data, dict)
        # Check for JAX arrays instead of NumPy arrays
        assert isinstance(list(train_data.condition_data.values())[0], jax.Array)
        assert train_data.max_combination_length == 1

        if sample_covariates == [] and perturbation_covariates == {"drug": ("drug1",)}:
            assert (
                train_data.n_perturbations
                == (len(adata_perturbation.obs["drug1"].cat.categories) - 1) * train_data.n_controls
            )
        # Check for JAX arrays instead of NumPy arrays
        assert isinstance(train_data.cell_data, jax.Array)
        assert isinstance(train_data.split_covariates_mask, jax.Array)
        assert isinstance(train_data.split_idx_to_covariates, dict)
        assert isinstance(train_data.perturbation_covariates_mask, jax.Array)
        assert isinstance(train_data.perturbation_idx_to_covariates, dict)
        assert isinstance(train_data.control_to_perturbation, dict)

    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariate_comb_args)
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_get_train_data_with_combinations(
        self,
        adata_perturbation: ad.AnnData,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
        algorithm,
    ):
        from cfp.data._datamanager import DataManager

        dm = DataManager(
            adata_perturbation,
            sample_rep="X",
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=["cell_type"],
            sample_covariate_reps={"cell_type": "cell_type"},
        )

        train_data = dm.get_train_data(adata_perturbation, algorithm=algorithm)

        assert ((train_data.perturbation_covariates_mask == -1) + (train_data.split_covariates_mask == -1)).all()

        if split_covariates == []:
            assert train_data.n_controls == 1
        if split_covariates == ["cell_type"]:
            assert train_data.n_controls == len(adata_perturbation.obs["cell_type"].cat.categories)

        assert isinstance(train_data.condition_data, dict)
        # Check for JAX arrays instead of NumPy arrays
        assert isinstance(list(train_data.condition_data.values())[0], jax.numpy.ndarray)
        assert train_data.max_combination_length == len(perturbation_covariates["drug"])

        for k in perturbation_covariates.keys():
            assert k in train_data.condition_data.keys()
            assert train_data.condition_data[k].ndim == 3
            assert train_data.condition_data[k].shape[1] == train_data.max_combination_length
            assert train_data.condition_data[k].shape[0] == train_data.n_perturbations

        for k, v in perturbation_covariate_reps.items():
            assert k in train_data.condition_data.keys()
            assert train_data.condition_data[v].shape[1] == train_data.max_combination_length
            assert train_data.condition_data[v].shape[0] == train_data.n_perturbations
            cov_key = perturbation_covariates[v][0]
            if cov_key == "drug_a":
                cov_name = cov_key
            else:
                cov_name = adata_perturbation.obs[cov_key].values[0]
            assert train_data.condition_data[v].shape[2] == adata_perturbation.uns[k][cov_name].shape[0]

        # Check for JAX arrays instead of NumPy arrays
        assert isinstance(train_data.cell_data, jax.numpy.ndarray)
        assert isinstance(train_data.split_covariates_mask, jax.numpy.ndarray)
        assert isinstance(train_data.split_idx_to_covariates, dict)
        assert isinstance(train_data.perturbation_covariates_mask, jax.numpy.ndarray)
        assert isinstance(train_data.perturbation_idx_to_covariates, dict)
        assert isinstance(train_data.control_to_perturbation, dict)

    @pytest.mark.parametrize("max_combination_length", [0, 4])
    def test_max_combination_length(self, adata_perturbation, max_combination_length):
        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = "control"
        perturbation_covariates = {"drug": ["drug1"]}
        perturbation_covariate_reps = {"drug": "drug"}

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            max_combination_length=max_combination_length,
        )

        train_data = dm.get_train_data(adata_perturbation)

        assert ((train_data.perturbation_covariates_mask == -1) + (train_data.split_covariates_mask == -1)).all()

        expected_max_combination_length = max(max_combination_length, len(perturbation_covariates["drug"]))
        assert dm._max_combination_length == expected_max_combination_length
        assert train_data.condition_data["drug"].shape[1] == expected_max_combination_length


class TestValidationData:
    @pytest.mark.parametrize("sample_rep", ["X", "X_pca"])
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariate_comb_args)
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_get_validation_data(
        self,
        adata_perturbation: ad.AnnData,
        sample_rep,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
        algorithm,
    ):
        from cfp.data._datamanager import DataManager

        control_key = "control"
        sample_covariates = ["cell_type"]
        sample_covariate_reps = {"cell_type": "cell_type"}

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
        )

        val_data = dm.get_validation_data(adata_perturbation, algorithm=algorithm)

        assert isinstance(val_data.cell_data, jax.Array)
        assert isinstance(val_data.split_covariates_mask, jax.Array)
        assert isinstance(val_data.split_idx_to_covariates, dict)
        assert isinstance(val_data.perturbation_covariates_mask, jax.Array)
        assert isinstance(val_data.perturbation_idx_to_covariates, dict)
        assert isinstance(val_data.control_to_perturbation, dict)
        assert val_data.max_combination_length == len(perturbation_covariates["drug"])

        assert isinstance(val_data.condition_data, dict)
        assert isinstance(list(val_data.condition_data.values())[0], jax.Array)

        if sample_covariates == [] and perturbation_covariates == {"drug": ("drug1",)}:
            assert (
                val_data.n_perturbations
                == (len(adata_perturbation.obs["drug1"].cat.categories) - 1) * val_data.n_controls
            )

    @pytest.mark.skip(reason="To discuss: why should it raise an error?")
    def test_raises_wrong_max_combination_length(self, adata_perturbation, algorithm):
        from cfp.data._datamanager import DataManager

        max_combination_length = 3
        adata = adata_perturbation
        sample_rep = "X"
        split_covariates = ["cell_type"]
        control_key = "control"
        perturbation_covariates = {"drug": ["drug1"]}
        perturbation_covariate_reps = {"drug": "drug"}

        with pytest.raises(
            ValueError,
            match=r".*max_combination_length.*",
        ):
            dm = DataManager(
                adata,
                sample_rep=sample_rep,
                split_covariates=split_covariates,
                control_key=control_key,
                perturbation_covariates=perturbation_covariates,
                perturbation_covariate_reps=perturbation_covariate_reps,
                max_combination_length=max_combination_length,
            )

            _ = dm.get_validation_data(adata_perturbation, algorithm=algorithm)


class TestPredictionData:
    @pytest.mark.parametrize("sample_rep", ["X", "X_pca"])
    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariate_comb_args)
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_get_prediction_data(
        self,
        adata_perturbation: ad.AnnData,
        algorithm: str,
        sample_rep,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
    ):
        from cfp.data._datamanager import DataManager

        control_key = "control"
        sample_covariates = ["cell_type"]
        sample_covariate_reps = {"cell_type": "cell_type"}

        dm = DataManager(
            adata_perturbation,
            sample_rep=sample_rep,
            split_covariates=split_covariates,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
        )

        adata_pred = adata_perturbation[:50].copy()
        adata_pred.obs["control"] = True
        pred_data = dm.get_prediction_data(
            adata_pred,
            covariate_data=adata_pred.obs,
            sample_rep=sample_rep,
            algorithm=algorithm,
        )

        assert isinstance(pred_data.cell_data, jax.Array)
        assert isinstance(pred_data.split_covariates_mask, jax.Array)
        assert isinstance(pred_data.split_idx_to_covariates, dict)
        assert isinstance(pred_data.perturbation_idx_to_covariates, dict)
        assert isinstance(pred_data.control_to_perturbation, dict)
        assert pred_data.max_combination_length == len(perturbation_covariates["drug"])

        assert isinstance(pred_data.condition_data, dict)
        assert isinstance(list(pred_data.condition_data.values())[0], jax.Array)

        if sample_covariates == [] and perturbation_covariates == {"drug": ("drug1",)}:
            assert (
                pred_data.n_perturbations
                == (len(adata_perturbation.obs["drug1"].cat.categories) - 1) * pred_data.n_controls
            )

    @pytest.mark.parametrize("split_covariates", [[], ["cell_type"]])
    @pytest.mark.parametrize("perturbation_covariates", perturbation_covariate_comb_args)
    @pytest.mark.parametrize("perturbation_covariate_reps", [{}, {"drug": "drug"}])
    def test_algorithm_equivalence(
        self,
        adata_perturbation: ad.AnnData,
        split_covariates,
        perturbation_covariates,
        perturbation_covariate_reps,
    ):
        """Test that both implementations produce equivalent results."""
        from cfp.data._datamanager import DataManager
        import numpy as np
        import warnings

        # Filter AnnData deprecation warnings for clearer test output
        warnings.filterwarnings("ignore", message="Importing read_.*from `anndata` is deprecated")

        # Create a fresh copy to avoid side effects
        adata = adata_perturbation.copy()

        # Create DataManager
        dm = DataManager(
            adata,
            sample_rep="X",
            split_covariates=split_covariates,
            control_key="control",
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=["cell_type"],
            sample_covariate_reps={"cell_type": "cell_type"},
        )

        # Get data using both implementations
        train_data_old = dm.get_train_data(adata, algorithm="old")
        train_data_mask = dm.get_train_data(adata, algorithm="mask")

        # Test equality of n_controls
        assert train_data_old.n_controls == train_data_mask.n_controls, (
            f"n_controls mismatch: old={train_data_old.n_controls}, mask={train_data_mask.n_controls}"
        )

        # Test equality of n_perturbations
        assert train_data_old.n_perturbations == train_data_mask.n_perturbations, (
            f"n_perturbations mismatch: old={train_data_old.n_perturbations}, mask={train_data_mask.n_perturbations}"
        )

        # Test the keys of control_to_perturbation are the same
        assert set(train_data_old.control_to_perturbation.keys()) == set(
            train_data_mask.control_to_perturbation.keys()
        ), "control_to_perturbation keys don't match"

        # Test the values of control_to_perturbation are the same
        for k in train_data_old.control_to_perturbation:
            old_ctrl = np.sort(train_data_old.control_to_perturbation[k])
            mask_ctrl = np.sort(train_data_mask.control_to_perturbation[k])
            np.testing.assert_array_equal(
                old_ctrl, mask_ctrl, err_msg=f"control_to_perturbation values don't match for key {k}"
            )

        # Test the masks are the same
        np.testing.assert_array_equal(
            train_data_old.split_covariates_mask,
            train_data_mask.split_covariates_mask,
            err_msg="split_covariates_mask doesn't match",
        )

        np.testing.assert_array_equal(
            train_data_old.perturbation_covariates_mask,
            train_data_mask.perturbation_covariates_mask,
            err_msg="perturbation_covariates_mask doesn't match",
        )

        # Test the mappings are the same
        assert train_data_old.split_idx_to_covariates == train_data_mask.split_idx_to_covariates, (
            "split_idx_to_covariates doesn't match"
        )

        assert set(train_data_old.perturbation_idx_to_covariates.keys()) == set(
            train_data_mask.perturbation_idx_to_covariates.keys()
        ), "perturbation_idx_to_covariates keys don't match"

        # Test the embeddings have the same shape and values
        for key in train_data_old.condition_data:
            assert train_data_old.condition_data[key].shape == train_data_mask.condition_data[key].shape, (
                f"condition_data[{key}] shape mismatch"
            )

            # Convert JAX arrays to numpy for comparison
            old_array = np.asarray(train_data_old.condition_data[key])
            mask_array = np.asarray(train_data_mask.condition_data[key])

            np.testing.assert_allclose(
                old_array, mask_array, rtol=1e-5, atol=1e-8, err_msg=f"condition_data[{key}] values don't match"
            )

        # Test that prediction data is also consistent
        adata_pred = adata[:30].copy()
        adata_pred.obs["control"] = True

        pred_old = dm.get_prediction_data(adata_pred, covariate_data=adata_pred.obs, sample_rep="X", algorithm="old")

        pred_mask = dm.get_prediction_data(adata_pred, covariate_data=adata_pred.obs, sample_rep="X", algorithm="mask")

        assert pred_old.n_controls == pred_mask.n_controls, (
            f"Prediction data n_controls mismatch: old={pred_old.n_controls}, mask={pred_mask.n_controls}"
        )

        # Test with validation data as well
        val_old = dm.get_validation_data(adata, algorithm="old")
        val_mask = dm.get_validation_data(adata, algorithm="mask")

        assert val_old.n_controls == val_mask.n_controls, (
            f"Validation data n_controls mismatch: old={val_old.n_controls}, mask={val_mask.n_controls}"
        )

    def test_edge_case_equivalence(self, adata_perturbation: ad.AnnData):
        """Test equivalence in edge cases."""
        from cfp.data._datamanager import DataManager
        import numpy as np

        # Create a copy for modification
        adata = adata_perturbation.copy()

        # Case 1: Dataset with very few controls
        few_controls = adata.copy()
        # Make most cells non-control
        few_controls.obs["control"] = False
        # Leave only 3 control cells
        few_controls.obs.iloc[:3, few_controls.obs.columns.get_loc("control")] = True

        dm1 = DataManager(
            few_controls,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1", "drug2"]},
            perturbation_covariate_reps={"drug": "drug"},
        )

        old_data1 = dm1.get_train_data(few_controls, algorithm="old")
        mask_data1 = dm1.get_train_data(few_controls, algorithm="mask")

        assert old_data1.n_controls == mask_data1.n_controls

        # Case 2: Dataset with all controls
        all_controls = adata.copy()
        all_controls.obs["control"] = True

        dm2 = DataManager(
            all_controls,
            sample_rep="X",
            split_covariates=["cell_type"],
            control_key="control",
            perturbation_covariates={"drug": ["drug1", "drug2"]},
        )

        old_data2 = dm2.get_train_data(all_controls, algorithm="old")
        mask_data2 = dm2.get_train_data(all_controls, algorithm="mask")

        assert old_data2.n_controls == mask_data2.n_controls
        assert old_data2.n_perturbations == mask_data2.n_perturbations

        # Case 3: Multiple split covariates
        dm3 = DataManager(
            adata,
            sample_rep="X",
            split_covariates=["cell_type", "dosage"],  # Multiple split covariates
            control_key="control",
            perturbation_covariates={"drug": ["drug1"]},
        )

        old_data3 = dm3.get_train_data(adata, algorithm="old")
        mask_data3 = dm3.get_train_data(adata, algorithm="mask")

        assert old_data3.n_controls == mask_data3.n_controls
