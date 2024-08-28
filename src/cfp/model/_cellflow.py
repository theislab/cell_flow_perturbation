import os
import types
from collections.abc import Callable, Sequence
from dataclasses import field as dc_field
from functools import partial
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
from ott.solvers import utils as solver_utils

from cfp import _constants
from cfp._types import Layers_separate_input_t, Layers_t
from cfp.data._data import BaseDataMixin, ValidationData
from cfp.data._dataloader import PredictionSampler, TrainSampler, ValidationSampler
from cfp.data._datamanager import DataManager
from cfp.networks._velocity_field import ConditionalVelocityField
from cfp.plotting import _utils
from cfp.solvers import _genot, _otfm
from cfp.training._callbacks import BaseCallback
from cfp.training._trainer import CellFlowTrainer

__all__ = ["CellFlow"]


class CellFlow:
    """CellFlow model for perturbation prediction using Flow Matching.

    Parameters
    ----------
        adata
            An :class:`~anndata.AnnData` object.
        solver
            Solver to use for training. Either "otfm" or "genot".
    """

    def __init__(self, adata: ad.AnnData, solver: Literal["otfm", "genot"] = "otfm"):

        self._adata = adata
        self._solver_class = _otfm.OTFlowMatching if solver == "otfm" else _genot.GENOT
        self._dataloader: TrainSampler | None = None
        self._trainer: CellFlowTrainer | None = None
        self._validation_data: dict[str, ValidationData] = {}
        self._solver: _otfm.OTFlowMatching | _genot.GENOT | None = None
        self._condition_dim: int | None = None
        self._vf: ConditionalVelocityField | None = None

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
        """Prepare dataloader for training from anndata object.

        Parameters
        ----------
            adata
                An :class:`~anndata.AnnData` object.
            sample_rep
                Key in :attr:`~anndata.AnnData.obsm` where the sample representation is stored or `X` to use `adata.X`.
            control_key
                Key of a boolean column in `adata.o` that defines the control samples.
            perturbation_covariates
                A dictionary where the keys indicate the name of the covariate group and the values are keys in `adata.obs`. The corresponding columns should be either boolean (presence/abscence of the perturbation) or numeric (concentration or magnitude of the perturbation). If multiple groups are provided, the first is interpreted as the primary perturbation and the others as covariates corresponding to these perturbations, e.g. `{"drug":("drugA", "drugB"), "time":("drugA_time", "drugB_time")}`.
            perturbation_covariate_reps
                A dictionary where the keys indicate the name of the covariate group and the values are keys in `adata.uns` storing a dictionary with the representation of the covariates. E.g. `{"drug":"drug_embeddings"}` with `adata.uns["drug_embeddings"] = {"drugA": np.array, "drugB": np.array}`.
            sample_covariates
                Keys in :attr:`~anndata.AnnData.obs` indicating sample covatiates to be taken into account for training and prediction, e.g. `["age", "cell_type"]`.
            sample_covariate_reps
                A dictionary where the keys indicate the name of the covariate group and the values are keys in `adata.uns` storing a dictionary with the representation of the covariates. E.g. `{"cell_type": "cell_type_embeddings"}` with `adata.uns["cell_type_embeddings"] = {"cell_typeA": np.array, "cell_typeB": np.array}`.
            split_covariates
                Covariates in :attr:`~anndata.AnnData.obs` to split all control cells into different control populations. The perturbed cells are also split according to these columns, but if these covariates should also be encoded in the model, the corresponding column should also be used in `perturbation_covariates` or `sample_covariates`.
            max_combination_length
                Maximum number of combinations of primary `perturbation_covariates`. If `None`, the value is inferred from the provided `perturbation_covariates`.
            null_value
                Value to use for padding to `max_combination_length`.

        Returns
        -------
            None

        """
        self.dm = DataManager(
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

        # TODO: rename to self.train_data
        self.train_data = self.dm.get_train_data(self.adata)

        self._data_dim = self.train_data.cell_data.shape[-1]

    def prepare_validation_data(
        self,
        adata: ad.AnnData,
        name: str,
        n_conditions_on_log_iteration: int | None = None,
        n_conditions_on_train_end: int | None = None,
    ) -> None:
        """Prepare validation data.

        Parameters
        ----------
            adata
                An :class:`~anndata.AnnData` object.
            name
                Name of the validation data.
            condition_id_key
                Key in :attr:`~anndata.AnnData.obs` or `covariate_data` indicating the condition name.
            n_conditions_on_log_iterations
                Number of conditions to use for computation callbacks at each logged iteration. If :obj:`None`, use all conditions.
            n_conditions_on_train_end
                Number of conditions to use for computation callbacks at the end of training. If :obj:`None`, use all conditions.

        Returns
        -------
            None
        """
        if self.train_data is None:
            raise ValueError(
                "Dataloader not initialized. Training data needs to be set up before preparing validation data. Please call prepare_data first."
            )
        val_data = self.dm.get_validation_data(
            adata,
            n_conditions_on_log_iteration=n_conditions_on_log_iteration,
            n_conditions_on_train_end=n_conditions_on_train_end,
        )
        self._validation_data[name] = val_data

    def prepare_model(
        self,
        encode_conditions: bool = True,
        condition_embedding_dim: int = 32,
        time_encoder_dims: Sequence[int] = (1024, 1024, 1024),
        time_encoder_dropout: float = 0.0,
        hidden_dims: Sequence[int] = (1024, 1024, 1024),
        hidden_dropout: float = 0.0,
        decoder_dims: Sequence[int] = (1024, 1024, 1024),
        decoder_dropout: float = 0.0,
        pooling: Literal[
            "mean", "attention_token", "attention_seed"
        ] = "attention_token",
        pooling_kwargs: dict[str, Any] = types.MappingProxyType({}),
        layers_before_pool: Layers_separate_input_t | Layers_t = dc_field(
            default_factory=lambda: []
        ),
        layers_after_pool: Layers_t = dc_field(default_factory=lambda: []),
        cond_output_dropout: float = 0.0,
        condition_encoder_kwargs: dict[str, Any] | None = None,
        pool_sample_covariates: bool = True,
        vf_act_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.silu,
        time_freqs: int = 1024,
        solver_kwargs: dict[str, Any] | None = None,
        flow: dict[Literal["constant_noise", "bridge"], float] | None = None,
        match_fn: Callable[[ArrayLike, ArrayLike], ArrayLike] = partial(
            solver_utils.match_linear, epsilon=0.1, scale_cost="mean"
        ),
        optimizer: optax.GradientTransformation = optax.adam(1e-4),
        seed=0,
    ) -> None:
        """Prepare model for training.

        Parameters
        ----------
            encode_conditions
                Whether to encode conditions.
            condition_embedding_dim
                Dimensions of the condition embedding.
            condition_encoder_kwargs
                Keyword arguments for the condition encoder.
            time_encoder_dims
                Dimensions of the time embedding.
            time_encoder_dropout
                Dropout rate for the time embedding.
            hidden_dims
                Dimensions of the hidden layers.
            hidden_dropout
                Dropout rate for the hidden layers.
            decoder_dims
                Dimensions of the output layers.
            pooling
                Pooling method.
            pooling_kwargs
                Keyword arguments for the pooling method.
            layers_before_pool
                Layers before pooling. Either a sequence of tuples with layer type and parameters or a dictionary with input-specific layers.
            layers_after_pool
                Layers after pooling.
            cond_output_dropout
                Dropout rate for the last layer of the condition encoder.
            condition_encoder_kwargs
                Keyword arguments for the condition encoder.
            pool_sample_covariates
                Whether to include sample covariates in the pooling.
            vf_act_fn
                Activation function of the velocity field.
            time_freqs
                Frequency of the cyclical time encoding.
            solver_kwargs
                Keyword arguments for the solver.
            decoder_dropout
                Dropout rate for the output layers.
            flow
                Flow to use for training. Should be a dict of the form `{"constant_noise": noise_val}` or `{"bridge": noise_val}`. If :obj:`None`, defaults to `{"constant_noise": 0.0}`.
            match_fn
                Matching function.
            optimizer
                Optimizer for training.
            seed
                Random seed.

        Returns
        -------
            None
        """
        flow = flow or {"constant_noise": 0.0}
        if self.train_data is None:
            raise ValueError(
                "Dataloader not initialized. Please call `prepare_data` first."
            )

        condition_encoder_kwargs = condition_encoder_kwargs or {}
        covariates_not_pooled = (
            [] if pool_sample_covariates else self.dm.sample_covariates
        )
        solver_kwargs = solver_kwargs or {}
        flow = flow or {"constant_noise": 0.0}

        self.vf = ConditionalVelocityField(
            output_dim=self._data_dim,
            max_combination_length=self.train_data.max_combination_length,
            encode_conditions=encode_conditions,
            condition_embedding_dim=condition_embedding_dim,
            covariates_not_pooled=covariates_not_pooled,
            pooling=pooling,
            pooling_kwargs=pooling_kwargs,
            layers_before_pool=layers_before_pool,
            layers_after_pool=layers_after_pool,
            cond_output_dropout=cond_output_dropout,
            condition_encoder_kwargs=condition_encoder_kwargs,
            act_fn=vf_act_fn,
            time_freqs=time_freqs,
            time_encoder_dims=time_encoder_dims,
            time_encoder_dropout=time_encoder_dropout,
            hidden_dims=hidden_dims,
            hidden_dropout=hidden_dropout,
            decoder_dims=decoder_dims,
            decoder_dropout=decoder_dropout,
        )

        flow, noise = next(iter(flow.items()))
        if flow == "constant_noise":
            flow = dynamics.ConstantNoiseFlow(noise)
        elif flow == "bridge":
            flow = dynamics.BrownianBridge(noise)
        else:
            raise NotImplementedError(
                f"The key of `flow` must be `'constant_noise'` or `'bridge'` but found {flow}."
            )

        if self._solver_class == _otfm.OTFlowMatching:
            self._solver = self._solver_class(
                vf=self.vf,
                match_fn=match_fn,
                flow=flow,
                optimizer=optimizer,
                conditions=self.train_data.condition_data,
                rng=jax.random.PRNGKey(seed),
                **solver_kwargs,
            )
        elif self._solver_class == _genot.GENOT:
            self._solver = self._solver_class(
                vf=self.vf,
                data_match_fn=match_fn,
                flow=flow,
                source_dim=self._data_dim,
                target_dim=self._data_dim,
                optimizer=optimizer,
                conditions=self.train_data.condition_data,
                rng=jax.random.PRNGKey(seed),
                **solver_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Solver must be an instance of OTFlowMatching or GENOT, got {type(self.solver)}"
            )
        self._trainer = CellFlowTrainer(solver=self.solver)  # type: ignore[arg-type]

    def train(
        self,
        num_iterations: int,
        batch_size: int = 64,
        valid_freq: int = 1000,
        callbacks: Sequence[BaseCallback] = [],
        monitor_metrics: Sequence[str] = [],
    ) -> None:
        """Train the model.

        Parameters
        ----------
            num_iterations
                Number of iterations to train the model.
            batch_size
                Batch size.
            valid_freq
                Frequency of validation.
            callbacks
                Callback functions.
            monitor_metrics
                Metrics to monitor.

        Returns
        -------
            None
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

        self._solver = self.trainer.train(
            dataloader=self._dataloader,
            num_iterations=num_iterations,
            valid_freq=valid_freq,
            valid_loaders=validation_loaders,
            callbacks=callbacks,
            monitor_metrics=monitor_metrics,
        )

    def predict(
        self,
        adata: ad.AnnData,
        sample_rep: str,
        covariate_data: pd.DataFrame,
        condition_id_key: str | None = None,
    ) -> dict[str, dict[str, ArrayLike]] | dict[str, ArrayLike]:
        """Predict perturbation.

        Parameters
        ----------
            adata
                An :class:`~anndata.AnnData` object with the source representation.
            sample_rep
                Key in :attr:`~anndata.AnnData.obsm` where the sample representation is stored or `X` to use `adata.X`.
            covariate_data
                Covariate data defining the condition to predict. If not provided, :attr:`~anndata.AnnData.obs` is used.
            condition_id_key
                Key in :attr:`~anndata.AnnData.obs` or `covariate_data` indicating the condition name.

        Returns
        -------
            A :class:`dict` with the predicted sample representation for each source distribution and condition.
        """
        if not self.solver.is_trained:  # type: ignore[union-attr]
            raise ValueError("Model not trained. Please call `train` first.")

        if adata is not None and covariate_data is not None:
            if self.dm.control_key not in adata.obs.columns:
                raise ValueError(
                    f"If both `adata` and `covariate_data` are given, the control key `{self.dm.control_key}` must be in `adata.obs`."
                )
            if not adata.obs[self.dm.control_key].all():
                raise ValueError(
                    f"If both `adata` and `covariate_data` are given, all samples in `adata` must be control samples, and thus `adata.obs[`{self.dm.control_key}`] must be set to `True` everywhere."
                )
        pred_data = self.dm.get_prediction_data(
            adata,
            sample_rep=sample_rep,
            covariate_data=covariate_data,
            condition_id_key=condition_id_key,
        )
        pred_loader = PredictionSampler(pred_data)
        batch = pred_loader.sample()
        src = batch["source"]
        condition = batch.get("condition", None)
        out = jax.tree.map(self.solver.predict, src, condition)  # type: ignore[union-attr]

        return out

    def get_condition_embedding(
        self,
        covariate_data: pd.DataFrame | BaseDataMixin | None = None,
        rep_dict: dict[str, str] | None = None,
        condition_id_key: str | None = None,
        key_added: str | None = _constants.CONDITION_EMBEDDING,
    ) -> pd.DataFrame:
        """Get condition embedding.

        Parameters
        ----------
            adata
                An :class:`~anndata.AnnData` object. If not provided, the training data is used.
            covariate_data
                Covariate data.
            condition_id_key
                Key in :attr:`anndata.AnnData.obs` or `covariate_data` indicating the condition name.
            key_added
                Key to store the condition embedding in :attr:`anndata.AnnData.uns`.

        Returns
        -------
            A :class:`~pandas.DataFrame` with the condition embeddings.
        """
        if self.solver is None:
            raise ValueError("Model not trained. Please call `train` first.")

        if not self.dm.is_conditional:
            raise ValueError(
                "Model is not conditional. Condition embeddings are not available."
            )

        if hasattr(covariate_data, "condition_data"):
            cond_data = covariate_data
        elif isinstance(covariate_data, pd.DataFrame):
            cond_data = self.dm.get_condition_data(
                covariate_data=covariate_data,
                rep_dict=rep_dict,
                condition_id_key=condition_id_key,
            )
        else:
            raise ValueError(
                "Covariate data must be a `pandas.DataFrame` or an instance of `BaseData`."
            )

        condition_embeddings: dict[str, ArrayLike] = {}
        n_conditions = len(next(iter(cond_data.condition_data.values())))  # type: ignore[union-attr]
        for i in range(n_conditions):
            condition = {k: v[[i], :] for k, v in cond_data.condition_data.items()}  # type: ignore[union-attr]
            if len(cond_data.perturbation_idx_to_id):  # type: ignore[union-attr]
                c_key = cond_data.perturbation_idx_to_id[i]  # type: ignore[union-attr]
            else:
                cov_combination = cond_data.perturbation_idx_to_covariates[i]  # type: ignore[union-attr]
                c_key = tuple(cov_combination[i] for i in range(len(cov_combination)))
            condition_embeddings[c_key] = self.solver.get_condition_embedding(condition)

        df = pd.DataFrame.from_dict(
            {k: v[0] for k, v in condition_embeddings.items()}  # type: ignore[index]
        ).T
        indices = list(self.dm.sample_covariates)
        for pert_cov in self.dm.perturbation_covariates:
            indices += [
                f"{pert_cov}_{i}" for i in range(self.dm.max_combination_length)
            ]
        df.index.set_names(indices, inplace=True)
        df.drop_duplicates(inplace=True)

        if key_added is not None:
            _utils.set_plotting_vars(self.adata, key=key_added, value=df)

        return df

    def save(
        self,
        dir_path: str,
        file_prefix: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Save the model. Pickles the CellFlow class instance.

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
    ):
        """
        Instantiate a CellFlow model from a saved output.

        Parameters
        ----------
            filename
                Path to the saved file

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
        """The AnnData object used for training."""
        return self._adata

    @property
    def solver(self) -> _otfm.OTFlowMatching | _genot.GENOT | None:
        """The trained solver."""
        return self._solver

    @property
    def dataloader(self) -> TrainSampler | None:
        """The dataloader used for training."""
        return self._dataloader

    @property
    def trainer(self) -> CellFlowTrainer | None:
        """The trainer used for training."""
        return self._trainer

    @property
    def validation_data(self) -> dict[str, ValidationData]:
        """The validation data."""
        return self._validation_data

    @property
    def velocity_field(self) -> ConditionalVelocityField | None:
        """The velocity field."""
        return self._vf

    @velocity_field.setter
    def velocity_field(self, vf: ConditionalVelocityField) -> None:
        """Set the velocity field."""
        if not isinstance(vf, ConditionalVelocityField):
            raise ValueError(
                f"Expected `vf` to be an instance of `ConditionalVelocityField`, found `{type(vf)}`."
            )
        self._vf = vf
