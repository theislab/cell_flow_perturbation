import jax
import jax.numpy as jnp
import optax
import pytest

import cfp

x_test = jnp.ones((10, 5)) * 10
t_test = jnp.ones((10, 1))
cond = {"pert1": jnp.ones((1, 2, 3))}


class TestVelocityField:
    @pytest.mark.parametrize("decoder_dims", [(32, 32), ()])
    @pytest.mark.parametrize("hidden_dims", [(32, 32), ()])
    @pytest.mark.parametrize("layer_norm_before_concatenation", [True, False])
    @pytest.mark.parametrize("linear_projection_before_concatenation", [True, False])
    @pytest.mark.parametrize("time_encoder_batchnorm", [True, False])
    @pytest.mark.parametrize("hidden_batchnorm", [True, False])
    @pytest.mark.parametrize("decoder_batchnorm", [True, False])
    def test_velocity_field_init(
        self,
        hidden_dims,
        decoder_dims,
        layer_norm_before_concatenation,
        linear_projection_before_concatenation,
        time_encoder_batchnorm,
        hidden_batchnorm,
        decoder_batchnorm,
    ):
        vf = cfp.networks.ConditionalVelocityField(
            output_dim=5,
            max_combination_length=2,
            condition_embedding_dim=12,
            hidden_dims=hidden_dims,
            decoder_dims=decoder_dims,
            layer_norm_before_concatenation=layer_norm_before_concatenation,
            linear_projection_before_concatenation=linear_projection_before_concatenation,
            time_encoder_batchnorm=time_encoder_batchnorm,
            hidden_batchnorm=hidden_batchnorm,
            decoder_batchnorm=decoder_batchnorm,
        )
        assert vf.output_dims == decoder_dims + (5,)

        vf_rng = jax.random.PRNGKey(111)
        opt = optax.adam(1e-3)
        vf_state = vf.create_train_state(vf_rng, opt, 5, cond, train=False)
        state_dict = {"params": vf_state.params}
        mutable = False
        # if time_encoder_batchnorm or hidden_batchnorm or decoder_batchnorm:
        if hasattr(vf_state, "batch_stats"):
            state_dict = {
                "params": vf_state.params,
                "batch_stats": vf_state.batch_stats,
            }
            mutable = ["batch_stats"]

        x_out = vf_state.apply_fn(
            state_dict, t_test, x_test, cond, train=True, mutable=mutable
        )
        # if time_encoder_batchnorm or hidden_batchnorm or decoder_batchnorm:
        if hasattr(vf_state, "batch_stats"):
            x_out, batch_stats = x_out
        assert x_out.shape == (10, 5)

        cond_embed = vf.apply(
            state_dict, cond, method="get_condition_embedding", mutable=mutable
        )
        # if time_encoder_batchnorm or hidden_batchnorm or decoder_batchnorm:
        if hasattr(vf_state, "batch_stats"):
            cond_embed, batch_stats = cond_embed
        assert cond_embed.shape == (1, 12)
