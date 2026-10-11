import numpy as np
import pandas as pd

from src.research_extensions import (
    causal_ema_windows,
    EmaWindowRidge,
    fit_online_onset_detector,
    history_windows,
    onset_detection_report,
    onset_probability,
    select_asymmetric_ridge,
    select_training_features,
    PolynomialWindowRidge,
    SummaryMLP,
)


def test_causal_ema_does_not_change_past_when_future_changes():
    sequence = np.arange(12, dtype=np.float32).reshape(1, 6, 2)
    changed_future = sequence.copy()
    changed_future[:, 4:, :] += 1000
    first = causal_ema_windows(sequence, 0.3)
    second = causal_ema_windows(changed_future, 0.3)
    np.testing.assert_array_equal(first[:, :4], second[:, :4])


def test_feature_selector_keeps_settings_and_ranks_using_only_given_training_windows():
    rng = np.random.default_rng(3)
    targets = np.linspace(0, 100, 80)
    sequences = rng.normal(size=(80, 5, 3)).astype(np.float32)
    sequences[:, :, 0] += targets[:, None] / 100
    indices, ranking = select_training_features(
        sequences, targets, ["sensor_1", "sensor_2", "setting_1"], top_sensors=1
    )
    assert indices == [0, 2]
    assert ranking.loc[ranking["feature"] == "sensor_1", "selected"].item()


def test_short_history_uses_only_prefix_and_left_pads_earliest_observation():
    frame = pd.DataFrame(
        {
            "unit": [1] * 4,
            "cycle": [1, 2, 3, 4],
            "sensor_1": [10.0, 20.0, 30.0, 40.0],
            "RUL": [3.0, 2.0, 1.0, 0.0],
        }
    )
    windows, units, cycles, targets = history_windows(
        frame, ["sensor_1"], model_window=4, observed_history=2
    )
    np.testing.assert_array_equal(windows[0, :, 0], [10.0, 10.0, 10.0, 20.0])
    np.testing.assert_array_equal(windows[-1, :, 0], [30.0, 30.0, 30.0, 40.0])
    np.testing.assert_array_equal(cycles, [2, 3, 4])
    np.testing.assert_array_equal(targets, [2.0, 1.0, 0.0])
    np.testing.assert_array_equal(units, [1, 1, 1])


def test_asymmetric_weight_selection_uses_validation_phm_score():
    rng = np.random.default_rng(9)
    train_x = rng.normal(size=(120, 8, 2)).astype(np.float32)
    train_y = np.maximum(0, 70 - train_x[:, -1, 0] * 12)
    validation_x = rng.normal(size=(60, 8, 2)).astype(np.float32)
    validation_y = np.maximum(0, 70 - validation_x[:, -1, 0] * 12)
    selected, weight, rows = select_asymmetric_ridge(
        train_x, train_y, validation_x, validation_y
    )
    assert weight in {0.5, 1.0, 2.0}
    assert len(rows) == 3
    assert selected.predict(validation_x).shape == (60,)


def test_engineered_model_wrappers_accept_shared_sequence_inputs():
    rng = np.random.default_rng(44)
    sequences = rng.normal(size=(40, 6, 4)).astype(np.float32)
    targets = np.linspace(0, 100, len(sequences))
    models = [
        EmaWindowRidge(0.35).fit(sequences, targets),
        PolynomialWindowRidge([0, 1, 2, 3]).fit(sequences, targets),
        SummaryMLP(seed=4, max_iter=3).fit(sequences, targets),
    ]
    assert all(model.predict(sequences[:5]).shape == (5,) for model in models)


def test_online_onset_uses_sensor_windows_and_reports_validation_delay():
    rng = np.random.default_rng(18)
    target = np.tile(np.r_[np.full(20, 125.0), np.linspace(120, 0, 30)], 3)
    sensor = (target < 125).astype(float)
    sequences = rng.normal(0, 0.02, size=(len(target), 6, 2)).astype(np.float32)
    sequences[:, :, 0] += sensor[:, None]
    sequences[:, :, 1] += sensor[:, None] * 0.5
    units = np.repeat(np.arange(1, 4), 50)
    cycles = np.tile(np.arange(1, 51), 3)
    detector = fit_online_onset_detector(sequences[:100], target[:100], rul_cap=125)
    probabilities = onset_probability(detector, sequences[100:])
    report, engines = onset_detection_report(
        units[100:], cycles[100:], target[100:], probabilities, rul_cap=125
    )
    assert probabilities.shape == (50,)
    assert report["n_engines"] == 1
    assert "median_detection_delay_cycles" in report
    assert len(engines) == 1
