"""Integration tests for EDA functionality with real data."""
import pandas as pd
import numpy as np
from app import (
    get_numeric_columns,
    get_categorical_columns
)


def test_sample_data():
    """Test EDA functions with the actual sample data."""
    print("Test: Sample Data Loading and Analysis")
    
    # Load sample data
    df = pd.read_csv('sample_data.csv')
    print(f"  Loaded sample_data.csv: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Test column detection
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    print(f"  Numeric columns: {numeric_cols}")
    print(f"  Categorical columns: {categorical_cols}")
    
    # Verify expected columns
    assert 'id' in numeric_cols, "id should be numeric"
    assert 'age' in numeric_cols, "age should be numeric"
    assert 'salary' in numeric_cols, "salary should be numeric"
    assert 'name' in categorical_cols, "name should be categorical"
    assert 'city' in categorical_cols, "city should be categorical"
    
    # Test correlation calculation
    corr = df[numeric_cols].corr()
    print(f"  Correlation matrix shape: {corr.shape}")
    assert corr.shape[0] == len(numeric_cols), "Correlation matrix size mismatch"
    
    # Test missing data
    missing = df.isnull().sum()
    print(f"  Total missing values: {missing.sum()}")
    
    print("  ✓ Passed\n")


def test_sample_data_with_missing():
    """Test EDA functions with sample data containing missing values."""
    print("Test: Sample Data With Missing Values")
    
    # Load sample data with missing values
    df = pd.read_csv('sample_data_with_missing.csv')
    print(f"  Loaded sample_data_with_missing.csv: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Test column detection
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    print(f"  Numeric columns: {numeric_cols}")
    print(f"  Categorical columns: {categorical_cols}")
    
    # Test missing data detection
    missing = df.isnull().sum()
    total_missing = missing.sum()
    print(f"  Total missing values: {total_missing}")
    print(f"  Missing by column:\n{missing[missing > 0]}")
    
    assert total_missing > 0, "Should have missing values in test data"
    
    # Test correlation with missing data
    corr = df[numeric_cols].corr()
    print(f"  Correlation matrix computed successfully: {corr.shape}")
    
    print("  ✓ Passed\n")


def test_edge_cases():
    """Test edge cases."""
    print("Test: Edge Cases")
    
    # Test with single row
    df_single = pd.DataFrame({
        'a': [1],
        'b': ['x']
    })
    numeric_cols = get_numeric_columns(df_single)
    categorical_cols = get_categorical_columns(df_single)
    print(f"  Single row - Numeric: {numeric_cols}, Categorical: {categorical_cols}")
    assert len(numeric_cols) == 1, "Should detect 1 numeric column"
    assert len(categorical_cols) == 1, "Should detect 1 categorical column"
    
    # Test with all numeric
    df_numeric = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4.5, 5.5, 6.5],
        'c': [7, 8, 9]
    })
    numeric_cols = get_numeric_columns(df_numeric)
    categorical_cols = get_categorical_columns(df_numeric)
    print(f"  All numeric - Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
    assert len(numeric_cols) == 3, "Should detect 3 numeric columns"
    assert len(categorical_cols) == 0, "Should detect 0 categorical columns"
    
    # Test with all categorical
    df_categorical = pd.DataFrame({
        'a': ['x', 'y', 'z'],
        'b': ['p', 'q', 'r'],
        'c': ['m', 'n', 'o']
    })
    numeric_cols = get_numeric_columns(df_categorical)
    categorical_cols = get_categorical_columns(df_categorical)
    print(f"  All categorical - Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
    assert len(numeric_cols) == 0, "Should detect 0 numeric columns"
    assert len(categorical_cols) == 3, "Should detect 3 categorical columns"
    
    print("  ✓ Passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Running EDA Integration Tests")
    print("=" * 60 + "\n")
    
    try:
        test_sample_data()
        test_sample_data_with_missing()
        test_edge_cases()
        
        print("=" * 60)
        print("All integration tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
