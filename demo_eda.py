"""
Demo script to showcase EDA functionality without Streamlit.
This demonstrates the core logic of the EDA functions.
"""
import pandas as pd
import numpy as np
from app import (
    get_numeric_columns,
    get_categorical_columns
)


def demo_column_detection():
    """Demonstrate column type detection."""
    print("=" * 60)
    print("DEMO: Column Type Detection")
    print("=" * 60)
    
    df = pd.read_csv('sample_data_with_missing.csv')
    
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n🔢 Numeric Columns ({len(numeric_cols)}):")
    for col in numeric_cols:
        print(f"   - {col}")
    
    print(f"\n📝 Categorical Columns ({len(categorical_cols)}):")
    for col in categorical_cols:
        print(f"   - {col}")


def demo_missing_data_analysis():
    """Demonstrate missing data analysis."""
    print("\n\n" + "=" * 60)
    print("DEMO: Missing Data Analysis")
    print("=" * 60)
    
    df = pd.read_csv('sample_data_with_missing.csv')
    
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': null_counts.index,
        'Missing Count': null_counts.values,
        'Missing %': null_percentages.values
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        'Missing Count', 
        ascending=False
    )
    
    print(f"\n🔍 Total Cells: {df.shape[0] * df.shape[1]}")
    print(f"❌ Total Missing: {int(null_counts.sum())}")
    print(f"📊 Overall Missing %: {(null_counts.sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")
    
    print("\n📋 Missing Data by Column:")
    print(missing_df.to_string(index=False))


def demo_correlation_analysis():
    """Demonstrate correlation analysis."""
    print("\n\n" + "=" * 60)
    print("DEMO: Correlation Analysis")
    print("=" * 60)
    
    df = pd.read_csv('sample_data_with_missing.csv')
    
    numeric_cols = get_numeric_columns(df)
    print(f"\n🔢 Computing correlations for {len(numeric_cols)} numeric columns")
    
    corr_matrix = df[numeric_cols].corr()
    
    print("\n📊 Correlation Matrix:")
    print(corr_matrix.round(3).to_string())
    
    print("\n🔥 Strong Correlations (|r| > 0.5):")
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                strong_corr.append({
                    'Column 1': corr_matrix.columns[i],
                    'Column 2': corr_matrix.columns[j],
                    'Correlation': f"{corr_val:.3f}"
                })
    
    if strong_corr:
        for sc in strong_corr:
            print(f"   - {sc['Column 1']} ↔ {sc['Column 2']}: {sc['Correlation']}")
    else:
        print("   (No strong correlations found)")


def demo_summary_statistics():
    """Demonstrate summary statistics."""
    print("\n\n" + "=" * 60)
    print("DEMO: Summary Statistics")
    print("=" * 60)
    
    df = pd.read_csv('sample_data_with_missing.csv')
    
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    print("\n📊 Numeric Column Statistics:")
    for col in numeric_cols:
        if col != 'id':  # Skip ID column
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                print(f"\n   {col}:")
                print(f"      Mean: {valid_data.mean():.2f}")
                print(f"      Median: {valid_data.median():.2f}")
                print(f"      Std Dev: {valid_data.std():.2f}")
                print(f"      Min: {valid_data.min():.2f}")
                print(f"      Max: {valid_data.max():.2f}")
    
    print("\n📝 Categorical Column Statistics:")
    for col in categorical_cols:
        if col != 'name':  # Skip name column
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                print(f"\n   {col}:")
                print(f"      Unique Values: {valid_data.nunique()}")
                print(f"      Most Common: {valid_data.value_counts().index[0]}")
                print(f"      Frequency: {valid_data.value_counts().values[0]}")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "EDA MODULE DEMONSTRATION" + " " * 24 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        demo_column_detection()
        demo_missing_data_analysis()
        demo_correlation_analysis()
        demo_summary_statistics()
        
        print("\n\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("=" * 60)
        print("\nTo see interactive visualizations, run:")
        print("   streamlit run app.py")
        print("\nThen upload 'sample_data_with_missing.csv' and explore the EDA tab.")
        print()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
