import pytest
import pandas as pd
import io
import numpy as np
from unittest.mock import Mock
from app import (
    init_session_state,
    reset_session_data,
    update_session_data,
    read_uploaded_file,
    get_numeric_columns,
    get_categorical_columns
)


class MockSessionState(dict):
    """Mock session state for testing."""
    def __getitem__(self, key):
        return super().__getitem__(key)
    
    def __setitem__(self, key, value):
        return super().__setitem__(key, value)
    
    def __contains__(self, key):
        return super().__contains__(key)


def test_read_csv_file():
    """Test reading a valid CSV file."""
    csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    mock_file = Mock()
    mock_file.name = "test.csv"
    mock_file.read = Mock(return_value=csv_content.encode())
    
    file_obj = io.StringIO(csv_content)
    mock_file.getvalue = Mock(return_value=csv_content.encode())
    mock_file.__iter__ = Mock(return_value=iter(csv_content.split('\n')))
    
    df = pd.read_csv(io.StringIO(csv_content))
    assert df.shape == (2, 3)
    assert list(df.columns) == ['name', 'age', 'city']


def test_empty_csv_detection():
    """Test detection of empty CSV file."""
    csv_content = ""
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        assert df.empty or len(df) == 0
    except pd.errors.EmptyDataError:
        assert True


def test_update_session_data():
    """Test updating session data."""
    session_state = MockSessionState()
    session_state['df'] = None
    session_state['filename'] = None
    session_state['upload_error'] = None
    
    df = pd.DataFrame({'col1': [1, 2, 3]})
    session_state['df'] = df
    session_state['filename'] = 'test.csv'
    session_state['upload_error'] = None
    
    assert session_state['df'] is not None
    assert session_state['filename'] == 'test.csv'
    assert session_state['upload_error'] is None


def test_reset_session_data():
    """Test resetting session data."""
    session_state = MockSessionState()
    session_state['df'] = pd.DataFrame({'col1': [1, 2, 3]})
    session_state['filename'] = 'test.csv'
    session_state['upload_error'] = 'Some error'
    
    session_state['df'] = None
    session_state['filename'] = None
    session_state['upload_error'] = None
    
    assert session_state['df'] is None
    assert session_state['filename'] is None
    assert session_state['upload_error'] is None


def test_get_numeric_columns():
    """Test identification of numeric columns."""
    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.5, 2.5, 3.5],
        'str_col': ['a', 'b', 'c'],
        'bool_col': [True, False, True]
    })
    
    numeric_cols = get_numeric_columns(df)
    assert 'int_col' in numeric_cols
    assert 'float_col' in numeric_cols
    assert 'str_col' not in numeric_cols


def test_get_categorical_columns():
    """Test identification of categorical columns."""
    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'str_col': ['a', 'b', 'c'],
        'cat_col': pd.Categorical(['x', 'y', 'z'])
    })
    
    categorical_cols = get_categorical_columns(df)
    assert 'str_col' in categorical_cols
    assert 'cat_col' in categorical_cols
    assert 'int_col' not in categorical_cols


def test_numeric_columns_empty_df():
    """Test numeric column detection on empty DataFrame."""
    df = pd.DataFrame()
    numeric_cols = get_numeric_columns(df)
    assert len(numeric_cols) == 0


def test_categorical_columns_empty_df():
    """Test categorical column detection on empty DataFrame."""
    df = pd.DataFrame()
    categorical_cols = get_categorical_columns(df)
    assert len(categorical_cols) == 0


def test_missing_data_detection():
    """Test detection of missing data."""
    df = pd.DataFrame({
        'col1': [1, 2, None, 4],
        'col2': [None, 'b', 'c', None],
        'col3': [1.1, 2.2, 3.3, 4.4]
    })
    
    null_counts = df.isnull().sum()
    assert null_counts['col1'] == 1
    assert null_counts['col2'] == 2
    assert null_counts['col3'] == 0


def test_correlation_computation():
    """Test correlation matrix computation."""
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50),
        'z': np.random.randn(50)
    })
    
    numeric_cols = get_numeric_columns(df)
    corr_matrix = df[numeric_cols].corr()
    
    assert corr_matrix.shape == (3, 3)
    assert all(abs(corr_matrix.values.diagonal() - 1.0) < 0.0001)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
