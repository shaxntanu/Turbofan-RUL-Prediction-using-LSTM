import numpy as np
import pandas as pd

from src.reliability import (
    leave_one_subset_sources,
    phm08_contributions,
    risk_coverage_rows,
    sensor_corruption_scenarios,
    terminal_windows,
    training_augmentation,
    validation_abstention_thresholds,
    validation_censor_windows,
)


def _labeled_frame() -> pd.DataFrame:
    rows = []
    for unit, length in ((1, 12), (2, 9)):
        for cycle in range(1, length + 1):
            rows.append(
                {
                    "unit": unit,
                    "cycle": cycle,
                    "RUL": length - cycle,
                    "sensor_1": float(cycle + unit),
                    "sensor_2": float(cycle * 0.5),
                    "sensor_3": float(unit - cycle * 0.1),
                }
            )
    return pd.DataFrame(rows)


def test_phm08_contributions_keep_asymmetric_early_and_late_penalties():
    values = phm08_contributions([20, 20], [7, 30])
    np.testing.assert_allclose(values, [np.e - 1, np.e - 1])


def test_leave_one_subset_out_excludes_target_from_every_source_list():
    datasets = ("FD001", "FD002", "FD003", "FD004")
    for target in datasets:
        sources = leave_one_subset_sources(datasets, target)
        assert len(sources) == len(datasets) - 1
        assert target not in sources


def test_terminal_windows_return_only_final_complete_window_per_engine():
    frame = _labeled_frame()
    sequences, units, cycles, targets = terminal_windows(frame, ["sensor_1"], 5)

    assert sequences.shape == (2, 5, 1)
    np.testing.assert_array_equal(units, [1, 2])
    np.testing.assert_array_equal(cycles, [12, 9])
    np.testing.assert_array_equal(targets, [0, 0])
    np.testing.assert_array_equal(sequences[0, :, 0], [9, 10, 11, 12, 13])


def test_validation_censor_windows_are_one_per_engine_and_reproducible():
    frame = _labeled_frame()
    first = validation_censor_windows(frame, ["sensor_1"], 5, seed=14)
    second = validation_censor_windows(frame, ["sensor_1"], 5, seed=14)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], [1, 2])
    assert len(first[1]) == frame["unit"].nunique()
    assert np.all(first[2] >= 5)


def test_corruption_scenarios_are_seeded_and_preserve_targets():
    frame = _labeled_frame()
    scenarios = sensor_corruption_scenarios(frame, ["sensor_1", "sensor_2", "sensor_3"], seed=81)
    assert any(scenario.kind == "individual_dropout" for scenario in scenarios)
    assert any(scenario.kind == "group_dropout" for scenario in scenarios)
    assert any(scenario.kind == "stuck" for scenario in scenarios)
    assert any(scenario.kind == "drift" for scenario in scenarios)
    assert any(scenario.kind == "noise" for scenario in scenarios)
    assert any(scenario.kind == "missing_block" for scenario in scenarios)
    assert any(scenario.kind == "reduced_sampling" for scenario in scenarios)

    for scenario in scenarios:
        first = scenario.apply(frame)
        second = scenario.apply(frame)
        keys = ["unit", "cycle"]
        expected = frame.set_index(keys)["RUL"].reindex(pd.MultiIndex.from_frame(first[keys]))
        np.testing.assert_array_equal(first["RUL"].to_numpy(), expected.to_numpy())
        pd.testing.assert_frame_equal(first, second)


def test_training_augmentation_is_deterministic_and_keeps_labels():
    rng = np.random.default_rng(3)
    sequences = rng.normal(size=(12, 6, 4)).astype(np.float32)
    targets = np.arange(12, dtype=np.float32)

    augmented_x, augmented_y = training_augmentation(sequences, targets, seed=9)
    repeated_x, repeated_y = training_augmentation(sequences, targets, seed=9)

    assert augmented_x.shape == (24, 6, 4)
    np.testing.assert_array_equal(augmented_x, repeated_x)
    np.testing.assert_array_equal(augmented_y, np.tile(targets, 2))
    assert not np.array_equal(augmented_x[12:], sequences)


def test_augmentation_ablation_modes_are_separately_configurable():
    rng = np.random.default_rng(31)
    sequences = rng.normal(size=(8, 5, 3)).astype(np.float32)
    targets = np.arange(8, dtype=np.float32)
    dropout_x, _ = training_augmentation(
        sequences, targets, seed=7, channel_dropout=0.5, noise_fraction=0.0
    )
    noise_x, _ = training_augmentation(
        sequences, targets, seed=7, channel_dropout=0.0, noise_fraction=0.1
    )

    np.testing.assert_array_equal(dropout_x[:8], sequences)
    np.testing.assert_array_equal(noise_x[:8], sequences)
    assert np.any(dropout_x[8:] == np.median(sequences, axis=(0, 1)))
    assert np.all(noise_x[8:] != sequences)


def test_abstention_thresholds_are_validation_derived_and_report_flag_counts():
    validation_scores = np.arange(10, dtype=float)
    thresholds = validation_abstention_thresholds(validation_scores, (0.0, 0.2))
    rows = risk_coverage_rows(
        np.arange(5, dtype=float),
        np.asarray([0, 2, 20, 5, 30], dtype=float),
        np.asarray([0, 0, 8, 1, 9], dtype=float),
        thresholds,
    )

    assert rows[0]["n_flagged"] == 0
    assert rows[1]["n_flagged"] == 2
    assert rows[1]["coverage_fraction"] == 0.6
    assert rows[1]["retained_mae"] == 1.0
