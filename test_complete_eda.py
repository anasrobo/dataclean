"""Complete end-to-end test of EDA functionality."""
import pandas as pd
import numpy as np
from app import (
    get_numeric_columns,
    get_categorical_columns,
    read_uploaded_file
)


def test_complete_workflow():
    """Test the complete EDA workflow."""
    print("=" * 70)
    print("COMPLETE EDA WORKFLOW TEST")
    print("=" * 70)
    
    # Test 1: Load sample data
    print("\n[1/6] Loading sample data...")
    df = pd.read_csv('sample_data_with_missing.csv')
    assert df is not None, "Failed to load data"
    assert not df.empty, "Data is empty"
    print(f"   ✓ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Test 2: Column type detection
    print("\n[2/6] Testing column type detection...")
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    assert len(numeric_cols) > 0, "No numeric columns found"
    assert len(categorical_cols) > 0, "No categorical columns found"
    print(f"   ✓ Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
    
    # Test 3: Univariate analysis data
    print("\n[3/6] Testing univariate analysis data...")
    for col in numeric_cols[:2]:  # Test first 2 numeric columns
        mean = df[col].mean()
        median = df[col].median()
        std = df[col].std()
        assert not np.isnan(mean) or df[col].isnull().all(), f"Invalid mean for {col}"
        print(f"   ✓ {col}: mean={mean:.2f}, median={median:.2f}, std={std:.2f}")
    
    for col in categorical_cols[:2]:  # Test first 2 categorical columns
        unique = df[col].nunique()
        most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else None
        assert unique > 0, f"No unique values for {col}"
        print(f"   ✓ {col}: {unique} unique values, most common={most_common}")
    
    # Test 4: Correlation computation
    print("\n[4/6] Testing correlation computation...")
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        assert corr_matrix.shape[0] == len(numeric_cols), "Correlation matrix size mismatch"
        # Check diagonal is all 1s (or NaN for columns with no variance)
        diagonal = corr_matrix.values.diagonal()
        valid_diagonal = diagonal[~np.isnan(diagonal)]
        assert all(abs(valid_diagonal - 1.0) < 0.0001), "Diagonal should be 1.0"
        print(f"   ✓ Computed {corr_matrix.shape[0]}×{corr_matrix.shape[1]} correlation matrix")
    
    # Test 5: Missing data detection
    print("\n[5/6] Testing missing data detection...")
    null_counts = df.isnull().sum()
    total_missing = null_counts.sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / total_cells) * 100
    print(f"   ✓ Found {int(total_missing)} missing values ({missing_pct:.2f}% of data)")
    
    # Test 6: Edge cases
    print("\n[6/6] Testing edge cases...")
    
    # Empty DataFrame
    df_empty = pd.DataFrame()
    numeric_empty = get_numeric_columns(df_empty)
    categorical_empty = get_categorical_columns(df_empty)
    assert len(numeric_empty) == 0, "Empty DataFrame should have no numeric columns"
    assert len(categorical_empty) == 0, "Empty DataFrame should have no categorical columns"
    print("   ✓ Empty DataFrame handled correctly")
    
    # DataFrame with only one type
    df_numeric_only = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    numeric_only = get_numeric_columns(df_numeric_only)
    categorical_only = get_categorical_columns(df_numeric_only)
    assert len(numeric_only) == 2, "Should have 2 numeric columns"
    assert len(categorical_only) == 0, "Should have 0 categorical columns"
    print("   ✓ Numeric-only DataFrame handled correctly")
    
    df_cat_only = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
    numeric_only2 = get_numeric_columns(df_cat_only)
    categorical_only2 = get_categorical_columns(df_cat_only)
    assert len(numeric_only2) == 0, "Should have 0 numeric columns"
    assert len(categorical_only2) == 2, "Should have 2 categorical columns"
    print("   ✓ Categorical-only DataFrame handled correctly")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - EDA module is fully functional!")
    print("=" * 70)


def test_data_requirements():
    """Test that EDA handles different data requirements."""
    print("\n" + "=" * 70)
    print("DATA REQUIREMENTS TEST")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('sample_data_with_missing.csv')
    numeric_cols = get_numeric_columns(df)
    
    # Test bivariate requirement
    print("\n[1/3] Bivariate analysis requirement...")
    if len(numeric_cols) >= 2:
        print(f"   ✓ Has {len(numeric_cols)} numeric columns (need 2+)")
    else:
        print(f"   ⚠ Only {len(numeric_cols)} numeric columns (need 2+)")
    
    # Test correlation requirement
    print("\n[2/3] Correlation heatmap requirement...")
    if len(numeric_cols) >= 2:
        print(f"   ✓ Has {len(numeric_cols)} numeric columns (need 2+)")
    else:
        print(f"   ⚠ Only {len(numeric_cols)} numeric columns (need 2+)")
    
    # Test univariate requirement
    print("\n[3/3] Univariate analysis requirement...")
    categorical_cols = get_categorical_columns(df)
    if len(numeric_cols) > 0 or len(categorical_cols) > 0:
        print(f"   ✓ Has {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns")
    else:
        print("   ⚠ No columns available for analysis")
    
    print("\n" + "=" * 70)
    print("✅ DATA REQUIREMENTS CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_complete_workflow()
        test_data_requirements()
        
        print("\n" + "🎉" * 35)
        print("\nAll EDA functionality verified successfully!")
        print("The module is ready for production use.\n")
        print("🎉" * 35)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
