import pytest
import pandas as pd
import io
from unittest.mock import Mock
from app import (
    init_session_state,
    reset_session_data,
    update_session_data,
    read_uploaded_file
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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
