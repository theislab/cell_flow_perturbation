import jax.tree_util as jtu
import numpy as np
import pytest

from cellflow.metrics import compute_mean_metrics, compute_metrics


class TestMetrics:
    @pytest.mark.parametrize("prefix", ["", "test_"])
    def test_compute_metrics(self, prefix):
        x_test = {
            "Alvespimycin+Pirarubicin": np.random.rand(50, 10),
            "Dacinostat+Danusertib": np.random.rand(50, 10),
        }
        y_test = {
            "Alvespimycin+Pirarubicin": np.random.rand(20, 10),
            "Dacinostat+Danusertib": np.random.rand(20, 10),
        }

        metrics = jtu.tree_map(compute_metrics, x_test, y_test)
        mean_metrics = compute_mean_metrics(metrics, prefix)

        assert "Alvespimycin+Pirarubicin" in metrics.keys()
        assert set(["r_squared", "sinkhorn_div_1", "sinkhorn_div_10", "sinkhorn_div_100", "e_distance", "mmd"]) <= set(
            metrics["Alvespimycin+Pirarubicin"].keys()
        )
        assert set(
            [
                prefix + "r_squared",
                prefix + "sinkhorn_div_1",
                prefix + "sinkhorn_div_10",
                prefix + "sinkhorn_div_100",
                prefix + "e_distance",
                prefix + "mmd",
            ]
        ) <= set(mean_metrics.keys())
