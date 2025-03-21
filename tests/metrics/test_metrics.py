import jax.tree_util as jtu
import pytest

import cellflow


class TestMetrics:
    @pytest.mark.parametrize("prefix", ["", "test_"])
    def test_compute_metrics(self, metrics_data, prefix):
        x_test = metrics_data["x_test"]
        y_test = metrics_data["y_test"]

        metrics = jtu.tree_map(cellflow.metrics.compute_metrics, x_test, y_test)
        mean_metrics = cellflow.metrics.compute_mean_metrics(metrics, prefix)

        assert "Alvespimycin+Pirarubicin" in metrics.keys()
        assert {"r_squared", "sinkhorn_div_1", "sinkhorn_div_10", "sinkhorn_div_100", "e_distance", "mmd"} == set(
            metrics["Alvespimycin+Pirarubicin"].keys()
        )
        assert {
            prefix + "r_squared",
            prefix + "sinkhorn_div_1",
            prefix + "sinkhorn_div_10",
            prefix + "sinkhorn_div_100",
            prefix + "e_distance",
            prefix + "mmd",
        } == set(mean_metrics.keys())

    @pytest.mark.parametrize("epsilon", [0.1, 1, 10])
    def test_function_output(self, metrics_data, epsilon):
        x_test = metrics_data["x_test"]["Alvespimycin+Pirarubicin"]
        y_test = metrics_data["y_test"]["Alvespimycin+Pirarubicin"]

        r_squared = cellflow.metrics.compute_r_squared(x_test, y_test)
        sinkhorn_div = cellflow.metrics.compute_sinkhorn_div(x_test, y_test, epsilon=epsilon)
        e_distance = cellflow.metrics.compute_e_distance(x_test, y_test)
        mmd = cellflow.metrics.compute_scalar_mmd(x_test, y_test)

        assert r_squared is not None
        assert r_squared <= 1
        assert sinkhorn_div is not None
        assert sinkhorn_div >= 0
        assert e_distance is not None
        assert e_distance >= 0
        assert mmd is not None
        assert mmd >= 0

    @pytest.mark.parametrize("gamma", [0.1, 1, 2])
    def test_fast_metrics(self, metrics_data, gamma):
        x_test = metrics_data["x_test"]["Alvespimycin+Pirarubicin"]
        y_test = metrics_data["y_test"]["Alvespimycin+Pirarubicin"]

        e_distance = cellflow.metrics.compute_e_distance(x_test, y_test)
        e_distance_fast = cellflow.metrics.compute_e_distance_fast(x_test, y_test)

        mmd = cellflow.metrics.maximum_mean_discrepancy(x_test, y_test, gamma, exact=True)
        mmd_fast = cellflow.metrics.maximum_mean_discrepancy(x_test, y_test, gamma, exact=False)

        assert e_distance_fast is not None
        assert abs(e_distance - e_distance_fast) < 1e-4
        assert mmd_fast is not None
        assert abs(mmd - mmd_fast) < 1e-5
