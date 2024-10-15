from collections.abc import Sequence
from typing import Any, Literal

import jax
import numpy as np
import jax.numpy as jnp
from numpy.typing import ArrayLike
from tqdm import tqdm

from cfp.networks._cfgen_ae import CFGenEncoder, CFGenDecoder

from cfp.external import NegativeBinomial
from cfp.data._dataloader import TrainSampler, ValidationSampler
from cfp.training._callbacks import BaseCallback, CallbackRunner

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
        encoder: CFGenEncoder,
        decoder: CFGenDecoder,
        seed: int = 0,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.modality_list = list(self.encoder.encoder_kwargs.keys())
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

        valid_pred_data: dict[str, dict[str, ArrayLike]] = {}
        valid_true_data: dict[str, dict[str, ArrayLike]] = {}
        for val_key, vdl in val_data.items():
            batch = vdl.sample(mode=mode)
            src = batch["source"]
            true_tgt = batch["counts"]
            valid_pred_data[val_key] = jax.tree.map(self.cfgen.predict, src, False)
            valid_true_data[val_key] = true_tgt

        return valid_true_data, valid_pred_data
    
    def _update_logs(self, logs: dict[str, Any]) -> None:
        """Update training logs."""
        for k, v in logs.items():
            if k not in self.training_logs:
                self.training_logs[k] = []

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

        def forward_pass(
                batch: dict[str, jnp.ndarray],
                training: bool
            ) -> float:
            """Performs a single optimization step"""
            ## retrieving size factors
            src = batch["src_cell_data"]
            x = {"rna": src}## hack, change asap
            size_factor = {}
            for mod in self.modality_list:
                size_factor[mod] = jnp.sum(x[mod], axis = 1, keepdims = True)
            ## need to do that stuff with the train state and shit
            z = self.encoder(x, training) ## AttributeError: "CFGenEncoder" object has no attribute "encoder"
            x_hat = self.decoder(z, size_factor, training) ## AttributeError: "CFGenDecoder" object has no attribute "decoder"
            loss = 0.0
            for mod in self.modality_list:
                if mod == "rna":
                    px = NegativeBinomial(mu=x_hat[mod], theta=jnp.exp(self.theta))
                    loss -= px.log_prob(x).sum(1).mean()
            return loss
            
        pbar = tqdm(range(num_iterations))
        for it in pbar:
            rng, rng_step_fn = jax.random.split(rng, 2)
            batch = dataloader.sample(rng)
            loss = forward_pass(batch, True)
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
