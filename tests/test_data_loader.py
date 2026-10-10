import pytest
import pandas as pd
import numpy as np
from src.config import ProjectConfig
from src.data_loader import DataLoader
from src.target_construction import TargetConstructor
from src.data_validator import DataValidator

@pytest.fixture
def config():
    return ProjectConfig()

@pytest.fixture
def loader(config):
    return DataLoader(config)

@pytest.fixture
def sample_train_df():
    # Create a dummy training dataset: 2 engines, sequential cycles
    data = {
        'unit': [1]*10 + [2]*5,
        'cycle': list(range(1, 11)) + list(range(1, 6)),
        'setting_1': [1.0]*15,
        'setting_2': [2.0]*15,
        'setting_3': [3.0]*15,
    }
    # Add 21 sensors
    for i in range(1, 22):
        data[f'sensor_{i}'] = np.random.rand(15)

    return pd.DataFrame(data)

@pytest.fixture
def sample_test_df():
    # Create a dummy test dataset: 2 engines, partial cycles
    data = {
        'unit': [1]*5 + [2]*3,
        'cycle': list(range(1, 6)) + list(range(1, 4)),
        'setting_1': [1.0]*8,
        'setting_2': [2.0]*8,
        'setting_3': [3.0]*8,
    }
    for i in range(1, 22):
        data[f'sensor_{i}'] = np.random.rand(8)
    return pd.DataFrame(data)

def test_data_loader_columns(loader, config):
    # Test if loader uses correct column names
    assert len(loader.columns) == 26
    assert loader.columns[0] == 'unit'
    assert loader.columns[1] == 'cycle'
    assert loader.columns[-1] == 'sensor_21'

def test_target_construction_train(sample_train_df):
    # Test training RUL calculation
    df_with_rul = TargetConstructor.compute_training_rul(sample_train_df)

    # Engine 1 max cycle is 10. Cycle 1 RUL should be 9. Cycle 10 RUL should be 0.
    unit1_df = df_with_rul[df_with_rul['unit'] == 1]
    assert unit1_df.iloc[0]['RUL'] == 9
    assert unit1_df.iloc[-1]['RUL'] == 0

    # Engine 2 max cycle is 5. Cycle 1 RUL should be 4. Cycle 5 RUL should be 0.
    unit2_df = df_with_rul[df_with_rul['unit'] == 2]
    assert unit2_df.iloc[0]['RUL'] == 4
    assert unit2_df.iloc[-1]['RUL'] == 0

def test_target_construction_test(sample_test_df):
    # Test test RUL calculation
    # Final RULs for Engine 1: 50, Engine 2: 30
    final_ruls = pd.Series([50, 30])
    df_with_rul = TargetConstructor.compute_test_rul(sample_test_df, final_ruls)

    # Engine 1: max_test_cycle=5, current=1, final=50 -> RUL = 5-1+50 = 54
    # Engine 1: max_test_cycle=5, current=5, final=50 -> RUL = 5-5+50 = 50
    unit1_df = df_with_rul[df_with_rul['unit'] == 1]
    assert unit1_df.iloc[0]['RUL'] == 54
    assert unit1_df.iloc[-1]['RUL'] == 50

    # Engine 2: max_test_cycle=3, current=1, final=30 -> RUL = 3-1+30 = 32
    # Engine 2: max_test_cycle=3, current=3, final=30 -> RUL = 3-3+30 = 30
    unit2_df = df_with_rul[df_with_rul['unit'] == 2]
    assert unit2_df.iloc[0]['RUL'] == 32
    assert unit2_df.iloc[-1]['RUL'] == 30

def test_data_validator_sequential(sample_train_df):
    # Valid case
    is_valid, issues = DataValidator.validate(sample_train_df, "FD001")
    assert is_valid
    assert len(issues) == 0

    # Invalid case: break sequence for unit 1
    invalid_df = sample_train_df.copy()
    invalid_df.loc[0, 'cycle'] = 99
    is_valid, issues = DataValidator.validate(invalid_df, "FD001")
    assert not is_valid
    assert any("Cycles are not sequential" in issue for issue in issues)

def test_data_validator_columns(sample_train_df):
    # Invalid case: remove a column
    invalid_df = sample_train_df.drop(columns=['sensor_21'])
    is_valid, issues = DataValidator.validate(invalid_df, "FD001")
    assert not is_valid
    assert any("Column mismatch" in issue for issue in issues)
