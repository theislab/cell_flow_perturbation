from collections.abc import Sequence
from typing import Any, Literal

import jax
import numpy as np
import jax.numpy as jnp
from numpy.typing import ArrayLike
from tqdm import tqdm

from cfp.model._cfgen import CFGen

from cfp.external import NegativeBinomial
from cfp.data._dataloader import TrainSampler, ValidationSampler
from cfp.training._callbacks import BaseCallback, CallbackRunner

from cfp._counts import normalize_expression


class CFGenAETrainer:
    """Trainer for the CFGen AutoEncoder with Negative Binomial noise model

    Parameters
    ----------
        dataloader
            Data sampler.
        ae
            CFGen AE architecture.
        seed
            Random seed for subsampling validation data.

    Returns
    -------
        None

    """

    def __init__(
        self,
        cfgen: CFGen,
        normalization_type: Literal[
            "none", "proportions", "log_gexp", "log_gexp_scaled"
        ] = "none",
        seed: int = 0,
    ):
        self.cfgen = cfgen
        self.normalization_type = normalization_type
        self.modality_list = list(self.cfgen.encoder.encoder_kwargs.keys())
        self.rng_subsampling = np.random.default_rng(seed)
        self.training_logs: dict[str, Any] = {}

    def _validation_step(
        self,
        val_data: dict[str, ValidationSampler],
        mode: Literal["on_log_iteration", "on_train_end"] = "on_log_iteration",
    ) -> tuple[
        dict[str, dict[str, ArrayLike]],
        dict[str, dict[str, ArrayLike]],
    ]:
        """Compute predictions for validation data."""
        # TODO: Sample fixed number of conditions to validate on
        rng = jax.random.PRNGKey(0)

        valid_pred_data: dict[str, dict[str, ArrayLike]] = {}
        valid_true_data: dict[str, dict[str, ArrayLike]] = {}
        for val_key, vdl in val_data.items():
            batch = vdl.sample(rng)
            counts = batch["src_cell_data"]
            valid_pred_data[val_key] = jax.tree.map(self.cfgen.predict, counts, False)
            valid_true_data[val_key] = counts

        return valid_true_data, valid_pred_data

    def _update_logs(self, logs: dict[str, Any]) -> None:
        """Update training logs."""
        for k, v in logs.items():
            if k not in self.training_logs:
                self.training_logs[k] = []
            self.training_logs[k].append(v)

    def train(
        self,
        dataloader: TrainSampler,
        num_iterations: int,
        valid_freq: int,
        valid_loaders: dict[str, ValidationSampler] | None = None,
        monitor_metrics: Sequence[str] = [],
        callbacks: Sequence[BaseCallback] = [],
    ) -> None:
        """Trains the model.

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
            The trained model.
        """
        self.training_logs = {"loss": []}
        rng = jax.random.PRNGKey(0)

        # Initiate callbacks
        valid_loaders = valid_loaders or {}
        crun = CallbackRunner(
            callbacks=callbacks,
        )
        crun.on_train_begin()

        pbar = tqdm(range(num_iterations))
        for it in pbar:
            rng, rng_step_fn = jax.random.split(rng, 2)
            batch = dataloader.sample(rng)
            loss = self.cfgen.fwd_fn(rng, batch, True)
            self.training_logs["loss"].append(float(loss))

            if ((it - 1) % valid_freq == 0) and (it > 1):
                # Get predictions from validation data
                valid_true_data, valid_pred_data = self._validation_step(
                    valid_loaders, mode="on_log_iteration"
                )

                # Run callbacks
                metrics = crun.on_log_iteration(valid_true_data, valid_pred_data)
                self._update_logs(metrics)

                # Update progress bar
                mean_loss = np.mean(self.training_logs["loss"][-valid_freq:])
                postfix_dict = {
                    metric: round(self.training_logs[metric][-1], 3)
                    for metric in monitor_metrics
                }
                postfix_dict["loss"] = round(mean_loss, 3)
                pbar.set_postfix(postfix_dict)

        if num_iterations > 0:
            valid_true_data, valid_pred_data = self._validation_step(
                valid_loaders, mode="on_train_end"
            )
            metrics = crun.on_train_end(valid_true_data, valid_pred_data)
            self._update_logs(metrics)
