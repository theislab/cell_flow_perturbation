import functools
import os
import types
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from typing import Any, Literal

import anndata as ad
import cloudpickle
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import pandas as pd
from numpy.typing import ArrayLike
from ott.neural.methods.flows import dynamics

from cfp import _constants
from cfp._logging import logger
from cfp._types import Layers_separate_input_t, Layers_t
from cfp.data._data import ConditionData, ValidationData
from cfp.data._dataloader import PredictionSampler, TrainSampler, ValidationSampler
from cfp.data._datamanager import DataManager
from cfp.model._utils import _write_predictions
from cfp.networks._cfgen_ae import CFGenEncoder, CFGenDecoder
from cfp.plotting import _utils
from cfp.solvers import _genot, _otfm
from cfp.training._callbacks import BaseCallback
from cfp.training._cfgen_ae_trainer import CFGenAETrainer
from cfp.utils import match_linear

__all__ = ["CellFlow"]


class CFGen:
    """"""

    def __init__(self, adata: ad.AnnData):
        self._adata = adata
        self._is_trained: bool = False
        self._dataloader: TrainSampler | None = None
        self._trainer: CFGenAETrainer | None = None
        self._encoder: CFGenEncoder | None = None
        self._decoder: CFGenDecoder | None = None
        self._validation_data: dict[str, ValidationData] = {}

    def prepare_data(
        self,
        sample_rep: str,
        control_key: str,
        perturbation_covariates: dict[str, Sequence[str]],
        perturbation_covariate_reps: dict[str, str] | None = None,
        sample_covariates: Sequence[str] | None = None,
        sample_covariate_reps: dict[str, str] | None = None,
        split_covariates: Sequence[str] | None = None,
        max_combination_length: int | None = None,
        null_value: float = 0.0,
    ) -> None:
        """Prepare the dataloader for training from :attr:`cfp.model.CellFlow.adata`.

        Parameters
        ----------
        sample_rep
            Key in :attr:`~anndata.AnnData.obsm` of :attr:`cfp.model.CellFlow.adata` where the sample representation is stored or ``'X'`` to use
            :attr:`~anndata.AnnData.X`.
        control_key
            Key of a boolean column in :attr:`~anndata.AnnData.obs` of :attr:`cfp.model.CellFlow.adata` that defines the control samples.
        perturbation_covariates
            A dictionary where the keys indicate the name of the covariate group and the values are
            keys in :attr:`~anndata.AnnData.obs` of :attr:`cfp.model.CellFlow.adata`. The corresponding columns can be of the following types:

            - categorial: The column contains categories whose representation is stored in
              :attr:`~anndata.AnnData.uns`, see ``'perturbation_covariate_reps'``.
            - boolean: The perturbation is present or absent.
            - numeric: The perturbation is given as a numeric value, possibly linked to a categorical
              perturbation, e.g. dosages for a drug.

            If multiple groups are provided, the first is interpreted as the primary
            perturbation and the others as covariates corresponding to these perturbations.
        perturbation_covariate_reps
            A :class:`dict` where the keys indicate the name of the covariate group and the values are
            keys in :attr:`~anndata.AnnData.uns` storing a dictionary with the representation of
            the covariates.
        sample_covariates
            Keys in :attr:`~anndata.AnnData.obs` indicating sample covariates. Sample covariates are
            defined such that each cell has only one value for each sample covariate (in constrast to
            ``'perturbation_covariates'`` which can have multiple values for each cell). If :obj:`None`, no sample
        sample_covariate_reps
            A dictionary where the keys indicate the name of the covariate group and the values
            are keys in :attr:`~anndata.AnnData.uns` storing a dictionary with the representation
            of the covariates.
        split_covariates
            Covariates in :attr:`~anndata.AnnData.obs` to split all control cells into different
            control populations. The perturbed cells are also split according to these columns,
            but if any of the ``'split_covariates'`` has a representation which should be incorporated by
            the model, the corresponding column should also be used in ``'perturbation_covariates'``.
        max_combination_length
            Maximum number of combinations of primary ``'perturbation_covariates'``. If :obj:`None`, the
            value is inferred from the provided ``'perturbation_covariates'`` as the maximal number of
            perturbations a cell has been treated with.
        null_value
            Value to use for padding to ``'max_combination_length'``.

        Returns
        -------
        Updates the following fields:

        - :attr:`cfp.model.CellFlow.dm` - the :class:`cfp.data.DataManager` object.
        - :attr:`cfp.model.CellFlow.train_data` - the training data.

        Example
        -------
            Consider the case where we have combinations of drugs along with dosages, saved in :attr:`anndata.AnnData.obs` as
            columns `drug_1` and `drug_2` with three different drugs `DrugA`, `DrugB`, and `DrugC`, and `dose_1` and `dose_2`
            for their dosages, respectively. We store the embeddings of the drugs in
            :attr:`anndata.AnnData.uns` under the key `drug_embeddings`, while the dosage columns are numeric. Moreover,
            we have a covariate `cell_type` with values `cell_typeA` and `cell_typeB`, with embeddings stored in
            :attr:`anndata.AnnData.uns` under the key `cell_type_embeddings`. Note that we then also have to set ``'split_covariates'``
            as we assume we have an unperturbed population for each cell type.

            .. code-block:: python

                perturbation_covariates = {{"drug": ("drug_1", "drug_2"), "dose": ("dose_1", "dose_2")}}
                perturbation_covariate_reps = {"drug": "drug_embeddings"}
                adata.uns["drug_embeddings"] = {
                    "drugA": np.array([0.1, 0.2, 0.3]),
                    "drugB": np.array([0.4, 0.5, 0.6]),
                    "drugC": np.array([-0.2, 0.3, 0.0]),
                }

                sample_covariates = {"cell_type": "cell_type_embeddings"}
                adata.uns["cell_type_embeddings"] = {
                    "cell_typeA": np.array([0.0, 1.0]),
                    "cell_typeB": np.array([0.0, 2.0]),
                }

                split_covariates = ["cell_type"]

                cf = CellFlow(adata)
                cf = cf.prepare_data(
                    sample_rep="X",
                    control_key="control",
                    perturbation_covariates=perturbation_covariates,
                    perturbation_covariate_reps=perturbation_covariate_reps,
                    sample_covariates=sample_covariates,
                    sample_covariate_reps=sample_covariate_reps,
                    split_covariates=split_covariates,
                )
        """
        self._dm = DataManager(
            self.adata,
            sample_rep=sample_rep,
            control_key=control_key,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariate_reps=perturbation_covariate_reps,
            sample_covariates=sample_covariates,
            sample_covariate_reps=sample_covariate_reps,
            split_covariates=split_covariates,
            max_combination_length=max_combination_length,
            null_value=null_value,
        )

        self.train_data = self._dm.get_train_data(self.adata)

        self._data_dim = self.train_data.cell_data.shape[-1]

    def prepare_validation_data(
        self,
        adata: ad.AnnData,
        name: str,
        n_conditions_on_log_iteration: int | None = None,
        n_conditions_on_train_end: int | None = None,
    ) -> None:
        """Prepare the validation data.

        Parameters
        ----------
        adata
            An :class:`anndata.AnnData` object.
        name
            Name of the validation data defining the key in :attr:`validation_data`.
        n_conditions_on_log_iterations
            Number of conditions to use for computation callbacks at each logged iteration.
            If :obj:`None`, use all conditions.
        n_conditions_on_train_end
            Number of conditions to use for computation callbacks at the end of training.
            If :obj:`None`, use all conditions.

        Returns
        -------
        :obj:`None`, and updates the following fields:

        - :attr:`cfp.model.CellFlow.validation_data` - a dictionary with the validation data.

        """
        if self.train_data is None:
            raise ValueError(
                "Dataloader not initialized. Training data needs to be set up before preparing validation data. Please call prepare_data first."
            )
        val_data = self._dm.get_validation_data(
            adata,
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
            n_conditions_on_train_end=n_conditions_on_train_end,
        )
        self._validation_data[name] = val_data

    def prepare_model(
        self,
        input_dim: dict[str, int],
        encoder_kwargs: dict[str, dict[str, Any]],
        covariate_specific_theta: bool,
        is_binarized: bool,
        encoder_multimodal_joint_layers: dict[str, Any] | None,
    ) -> None:
        """Initializes the CFGen Auto-Encoder architecture and its trainer"""
        self._encoder = CFGenEncoder(
            input_dim = input_dim,
            encoder_kwargs = encoder_kwargs,
            covariate_specific_theta = covariate_specific_theta,
            is_binarized = is_binarized,
            encoder_multimodal_joint_layers = encoder_multimodal_joint_layers
        )
        self._decoder = CFGenDecoder(
            input_dim = input_dim,
            encoder_kwargs = encoder_kwargs,
            covariate_specific_theta = covariate_specific_theta,
            is_binarized = is_binarized,
            encoder_multimodal_joint_layers = encoder_multimodal_joint_layers
        )
        self._trainer = CFGenAETrainer(self._encoder, self._decoder)  # type: ignore[arg-type]

    def train(
        self,
        num_iterations: int,
        batch_size: int = 1024,
        valid_freq: int = 1000,
        callbacks: Sequence[BaseCallback] = [],
        monitor_metrics: Sequence[str] = [],
    ) -> None:
        """Train the model.

        Note
        ----
        A low value of ``'valid_freq'`` results in long training
        because predictions are time-consuming compared to training steps.

        Parameters
        ----------
        num_iterations
            Number of iterations to train the model.
        batch_size
            Batch size.
        valid_freq
            Frequency of validation.
        callbacks
            Callbacks to perform at each validation step. There are two types of callbacks:
            - Callbacks for computations should inherit from :class:`cfp.training.ComputationCallback:` see e.g.
              :class:`cfp.training.Metrics`.
            - Callbacks for logging should inherit from :class:`~cfp.training.LoggingCallback` see e.g.
              :class:`~cfp.training.WandbLogger`.
        monitor_metrics
            Metrics to monitor.

        Returns
        -------
        Updates the following fields:

        - :attr:`cfp.model.CellFlow.dataloader` - the training dataloader.
        - :attr:`cfp.model.CellFlow.solver` - the trained solver.
        """
        if self.train_data is None:
            raise ValueError("Data not initialized. Please call `prepare_data` first.")

        if self.trainer is None:
            raise ValueError(
                "Model not initialized. Please call `prepare_model` first."
            )

        self._dataloader = TrainSampler(data=self.train_data, batch_size=batch_size)
        validation_loaders = {
            k: ValidationSampler(v) for k, v in self.validation_data.items()
        }

        self.trainer.train(
            dataloader=self._dataloader,
            num_iterations=num_iterations,
            valid_freq=valid_freq,
            valid_loaders=validation_loaders,
            callbacks=callbacks,
            monitor_metrics=monitor_metrics,
        )

    def save(
        self,
        dir_path: str,
        file_prefix: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save the model.

        Pickles the CellFlow object.

        Parameters
        ----------
            dir_path
                Path to a directory, defaults to current directory
            file_prefix
                Prefix to prepend to the file name.
            overwrite
                Overwrite existing data or not.

        Returns
        -------
            None
        """
        file_name = (
            f"{file_prefix}_{self.__class__.__name__}.pkl"
            if file_prefix is not None
            else f"{self.__class__.__name__}.pkl"
        )
        file_dir = (
            os.path.join(dir_path, file_name) if dir_path is not None else file_name
        )

        if not overwrite and os.path.exists(file_dir):
            raise RuntimeError(
                f"Unable to save to an existing file `{file_dir}` use `overwrite=True` to overwrite it."
            )
        with open(file_dir, "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(
        cls,
        filename: str,
    ) -> "CellFlow":
        """
        Load a :class:`cfp.model.CellFlow` model from a saved instance.

        Parameters
        ----------
            filename
                Path to the saved file.

        Returns
        -------
        Loaded instance of the model.
        """
        # Check if filename is a directory
        file_name = (
            os.path.join(filename, f"{cls.__name__}.pkl")
            if os.path.isdir(filename)
            else filename
        )

        with open(file_name, "rb") as f:
            model = cloudpickle.load(f)

        if type(model) is not cls:
            raise TypeError(
                f"Expected the model to be type of `{cls}`, found `{type(model)}`."
            )
        return model

    @property
    def adata(self) -> ad.AnnData:
        """The :class:`~anndata.AnnData` object used for training."""
        return self._adata

    @property
    def dataloader(self) -> TrainSampler | None:
        """The dataloader used for training."""
        return self._dataloader

    @property
    def trainer(self) -> CFGenAETrainer | None:
        """The trainer used for training."""
        return self._trainer

    @property
    def validation_data(self) -> dict[str, ValidationData]:
        """The validation data."""
        return self._validation_data

    @property
    def data_manager(self) -> DataManager:
        """The data manager, initialised with :attr:`cfp.model.CellFlow.adata`."""
        return self._dm
    
    @property
    def encoder(self) -> CFGenEncoder:
        """The encoder of the underlying CFGen Model"""
        return self._encoder
    
    @property
    def decoder(self) -> CFGenDecoder:
        """The encoder of the underlying CFGen Model"""
        return self._decoder