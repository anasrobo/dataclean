import streamlit as st
import pandas as pd
from typing import Optional
import io
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def init_session_state():
    """Initialize session state variables."""
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'filename' not in st.session_state:
        st.session_state['filename'] = None
    if 'upload_error' not in st.session_state:
        st.session_state['upload_error'] = None
    if 'cleaned_df' not in st.session_state:
        st.session_state['cleaned_df'] = None


def reset_session_data():
    """Reset all session data to initial state."""
    st.session_state['df'] = None
    st.session_state['filename'] = None
    st.session_state['upload_error'] = None
    st.session_state['cleaned_df'] = None


def update_session_data(df: Optional[pd.DataFrame], filename: Optional[str], error: Optional[str] = None):
    """
    Update session data with new DataFrame and metadata.
    
    Args:
        df: The pandas DataFrame to store
        filename: Name of the uploaded file
        error: Error message if upload/parsing failed
    """
    st.session_state['df'] = df
    st.session_state['filename'] = filename
    st.session_state['upload_error'] = error
    st.session_state['cleaned_df'] = df.copy() if df is not None else None


def read_uploaded_file(uploaded_file) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Read uploaded CSV or Excel file into a pandas DataFrame.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Tuple of (DataFrame or None, error_message or None)
    """
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            return None, f"Unsupported file format: .{file_extension}. Please upload a CSV or Excel file."
        
        if df.empty:
            return None, "The uploaded file is empty."
        
        return df, None
        
    except pd.errors.EmptyDataError:
        return None, "The file is empty or contains no valid data."
    except pd.errors.ParserError as e:
        return None, f"Failed to parse the file: {str(e)}"
    except Exception as e:
        return None, f"Error reading file: {str(e)}"


def render_data_preview(df: pd.DataFrame):
    """
    Render a preview of the loaded DataFrame including shape, dtypes, and head.
    
    Args:
        df: The pandas DataFrame to preview
    """
    st.subheader("📊 Data Preview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📋 Head", "🔢 Data Types", "📈 Summary Statistics"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with tab3:
        try:
            st.dataframe(df.describe(include='all'), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate summary statistics: {str(e)}")


def render_empty_state():
    """Render empty state messaging when no data is loaded."""
    st.info("👋 Welcome! Please upload a CSV or Excel file to get started.")
    
    st.markdown("""
    ### Supported File Formats
    - **CSV** (.csv)
    - **Excel** (.xlsx, .xls)
    
    ### Getting Started
    1. Click the "Browse files" button above
    2. Select your data file
    3. View the data preview and summary statistics
    """)


def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Get list of numeric columns from DataFrame.
    
    Args:
        df: The pandas DataFrame
        
    Returns:
        List of numeric column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list:
    """
    Get list of categorical columns from DataFrame.
    
    Args:
        df: The pandas DataFrame
        
    Returns:
        List of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def render_univariate_analysis(df: pd.DataFrame):
    """
    Render univariate analysis section with automatic plot type detection.
    
    Args:
        df: The pandas DataFrame to analyze
    """
    st.subheader("📊 Univariate Analysis")
    
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    if len(numeric_cols) == 0 and len(categorical_cols) == 0:
        st.warning("No numeric or categorical columns found for univariate analysis.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(numeric_cols) > 0:
            selected_numeric = st.selectbox(
                "Select numeric column",
                numeric_cols,
                key="univariate_numeric"
            )
        else:
            selected_numeric = None
            st.info("No numeric columns available")
    
    with col2:
        if len(categorical_cols) > 0:
            selected_categorical = st.selectbox(
                "Select categorical column",
                categorical_cols,
                key="univariate_categorical"
            )
        else:
            selected_categorical = None
            st.info("No categorical columns available")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        if selected_numeric:
            try:
                fig = px.histogram(
                    df,
                    x=selected_numeric,
                    title=f"Distribution of {selected_numeric}",
                    nbins=30,
                    marginal="box"
                )
                fig.update_layout(
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                with stats_col1:
                    st.metric("Mean", f"{df[selected_numeric].mean():.2f}")
                with stats_col2:
                    st.metric("Median", f"{df[selected_numeric].median():.2f}")
                with stats_col3:
                    st.metric("Std Dev", f"{df[selected_numeric].std():.2f}")
            except Exception as e:
                st.warning(f"Could not create histogram for {selected_numeric}: {str(e)}")
    
    with col_right:
        if selected_categorical:
            try:
                value_counts = df[selected_categorical].value_counts().head(20)
                
                if len(value_counts) == 0:
                    st.warning(f"No data available for {selected_categorical}")
                else:
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Distribution of {selected_categorical}",
                        labels={'x': selected_categorical, 'y': 'Count'}
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    stats_col1, stats_col2 = st.columns(2)
                    with stats_col1:
                        st.metric("Unique Values", df[selected_categorical].nunique())
                    with stats_col2:
                        st.metric("Most Common", str(value_counts.index[0]))
            except Exception as e:
                st.warning(f"Could not create bar chart for {selected_categorical}: {str(e)}")


def render_bivariate_analysis(df: pd.DataFrame):
    """
    Render bivariate analysis section with scatter and box plots.
    
    Args:
        df: The pandas DataFrame to analyze
    """
    st.subheader("🔍 Bivariate Analysis")
    
    numeric_cols = get_numeric_columns(df)
    all_cols = df.columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for bivariate analysis.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_axis = st.selectbox(
            "Select X-axis",
            all_cols,
            key="bivariate_x"
        )
    
    with col2:
        y_axis = st.selectbox(
            "Select Y-axis",
            numeric_cols,
            index=min(1, len(numeric_cols) - 1),
            key="bivariate_y"
        )
    
    with col3:
        plot_type = st.selectbox(
            "Select plot type",
            ["Scatter Plot", "Box Plot"],
            key="bivariate_plot_type"
        )
    
    try:
        if plot_type == "Scatter Plot":
            if x_axis not in numeric_cols:
                st.warning(f"X-axis column '{x_axis}' is not numeric. Scatter plots work best with numeric data.")
            
            color_col = st.selectbox(
                "Color by (optional)",
                ["None"] + all_cols,
                key="scatter_color"
            )
            
            color = None if color_col == "None" else color_col
            
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=color,
                title=f"{y_axis} vs {x_axis}",
                trendline="ols" if x_axis in numeric_cols else None
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            if x_axis in numeric_cols:
                correlation = df[[x_axis, y_axis]].corr().iloc[0, 1]
                st.metric("Correlation Coefficient", f"{correlation:.3f}")
        
        elif plot_type == "Box Plot":
            fig = px.box(
                df,
                x=x_axis,
                y=y_axis,
                title=f"{y_axis} by {x_axis}",
                points="outliers"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.warning(f"Could not create {plot_type.lower()}: {str(e)}")


def render_correlation_heatmap(df: pd.DataFrame):
    """
    Render correlation heatmap for numeric columns.
    
    Args:
        df: The pandas DataFrame to analyze
    """
    st.subheader("🔥 Correlation Heatmap")
    
    numeric_cols = get_numeric_columns(df)
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns to compute correlations.")
        return
    
    try:
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            xaxis_title="",
            yaxis_title="",
            height=max(400, len(numeric_cols) * 50),
            width=max(400, len(numeric_cols) * 50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Strong Correlations")
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
            st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
        else:
            st.info("No strong correlations (|r| > 0.5) found.")
            
    except Exception as e:
        st.warning(f"Could not compute correlation matrix: {str(e)}")


def render_missing_data_analysis(df: pd.DataFrame):
    """
    Render missing data visualization.
    
    Args:
        df: The pandas DataFrame to analyze
    """
    st.subheader("🔍 Missing Data Analysis")
    
    null_counts = df.isnull().sum()
    null_percentages = (null_counts / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': null_counts.index,
        'Missing Count': null_counts.values,
        'Missing Percentage': null_percentages.values
    })
    
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(
        'Missing Count', 
        ascending=False
    )
    
    if len(missing_df) == 0:
        st.success("✅ No missing data found in the dataset!")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            missing_df,
            x='Column',
            y='Missing Count',
            title='Missing Values by Column',
            labels={'Missing Count': 'Number of Missing Values'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            missing_df,
            x='Column',
            y='Missing Percentage',
            title='Missing Values Percentage by Column',
            labels={'Missing Percentage': 'Percentage (%)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Missing Data Summary")
    st.dataframe(missing_df.reset_index(drop=True), use_container_width=True)
    
    total_cells = df.shape[0] * df.shape[1]
    total_missing = null_counts.sum()
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Total Missing Values", int(total_missing))
    with metric_col2:
        st.metric("Columns with Missing Data", len(missing_df))
    with metric_col3:
        st.metric("Overall Missing %", f"{(total_missing / total_cells * 100):.2f}%")


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: The pandas DataFrame
        
    Returns:
        Tuple of (cleaned_df, num_duplicates_removed)
    """
    initial_rows = len(df)
    df_cleaned = df.drop_duplicates()
    num_removed = initial_rows - len(df_cleaned)
    return df_cleaned, num_removed


def fill_missing_values(df: pd.DataFrame, strategy: str, column: Optional[str] = None, constant_value: Optional[str] = None) -> tuple[pd.DataFrame, int]:
    """
    Fill missing values using specified strategy.
    
    Args:
        df: The pandas DataFrame
        strategy: 'drop', 'mean', 'median', 'mode', or 'constant'
        column: Column to apply strategy to (for numeric strategies)
        constant_value: Value to use for constant fill strategy
        
    Returns:
        Tuple of (cleaned_df, num_rows_affected)
    """
    df_copy = df.copy()
    
    if strategy == 'drop':
        initial_rows = len(df_copy)
        df_copy = df_copy.dropna()
        num_affected = initial_rows - len(df_copy)
    elif strategy == 'mean':
        if column and column in df_copy.columns and df_copy[column].dtype in [np.int64, np.float64, float, int]:
            mean_val = df_copy[column].mean()
            df_copy[column] = df_copy[column].fillna(mean_val)
            num_affected = df[column].isnull().sum()
        else:
            return df_copy, 0
    elif strategy == 'median':
        if column and column in df_copy.columns and df_copy[column].dtype in [np.int64, np.float64, float, int]:
            median_val = df_copy[column].median()
            df_copy[column] = df_copy[column].fillna(median_val)
            num_affected = df[column].isnull().sum()
        else:
            return df_copy, 0
    elif strategy == 'mode':
        if column and column in df_copy.columns:
            mode_val = df_copy[column].mode()
            if len(mode_val) > 0:
                df_copy[column] = df_copy[column].fillna(mode_val[0])
                num_affected = df[column].isnull().sum()
            else:
                return df_copy, 0
        else:
            return df_copy, 0
    elif strategy == 'constant':
        if column and column in df_copy.columns and constant_value is not None:
            df_copy[column] = df_copy[column].fillna(constant_value)
            num_affected = df[column].isnull().sum()
        else:
            return df_copy, 0
    else:
        return df_copy, 0
    
    return df_copy, int(num_affected)


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> tuple[pd.Series, float, float]:
    """
    Detect outliers using IQR method.
    
    Args:
        df: The pandas DataFrame
        column: Column name to detect outliers in
        
    Returns:
        Tuple of (outlier_mask, lower_bound, upper_bound)
    """
    if column not in df.columns or df[column].dtype not in [np.int64, np.float64, float, int]:
        return pd.Series([False] * len(df)), None, None
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return outlier_mask, lower_bound, upper_bound


def remove_outliers_iqr(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, int]:
    """
    Remove outliers using IQR method for selected columns.
    
    Args:
        df: The pandas DataFrame
        columns: List of column names to check for outliers
        
    Returns:
        Tuple of (cleaned_df, num_outliers_removed)
    """
    df_copy = df.copy()
    initial_rows = len(df_copy)
    
    combined_mask = pd.Series([False] * len(df_copy))
    
    for col in columns:
        outlier_mask, _, _ = detect_outliers_iqr(df_copy, col)
        combined_mask = combined_mask | outlier_mask
    
    df_copy = df_copy[~combined_mask]
    num_removed = initial_rows - len(df_copy)
    
    return df_copy, num_removed


def render_data_cleaning_tab(df: pd.DataFrame):
    """
    Render the Data Cleaning tab with interactive controls for data cleaning operations.
    
    Args:
        df: The pandas DataFrame to clean
    """
    if df is None or df.empty:
        st.info("👋 Please upload a data file to perform cleaning operations.")
        return
    
    st.header("🧹 Data Cleaning")
    
    if 'cleaned_df' not in st.session_state:
        st.session_state['cleaned_df'] = df.copy()
    
    cleaning_tabs = st.tabs([
        "🔄 Remove Duplicates",
        "❌ Handle Missing Values",
        "📊 Detect & Remove Outliers"
    ])
    
    with cleaning_tabs[0]:
        st.subheader("🔄 Remove Duplicate Rows")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Rows", len(st.session_state['cleaned_df']))
        
        with col2:
            num_duplicates = len(st.session_state['cleaned_df']) - len(st.session_state['cleaned_df'].drop_duplicates())
            st.metric("Duplicate Rows", num_duplicates)
        
        if num_duplicates > 0:
            if st.button("Remove Duplicates", key="remove_duplicates_btn", use_container_width=True):
                cleaned, num_removed = remove_duplicates(st.session_state['cleaned_df'])
                st.session_state['cleaned_df'] = cleaned
                st.success(f"✅ Removed {num_removed} duplicate row(s). New row count: {len(cleaned)}")
                st.rerun()
        else:
            st.info("✨ No duplicate rows found in the dataset.")
    
    with cleaning_tabs[1]:
        st.subheader("❌ Handle Missing Values")
        
        null_counts = st.session_state['cleaned_df'].isnull().sum()
        total_missing = null_counts.sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Missing Values", int(total_missing))
        with col2:
            st.metric("Columns with Missing", int((null_counts > 0).sum()))
        with col3:
            st.metric("Current Rows", len(st.session_state['cleaned_df']))
        
        if total_missing == 0:
            st.success("✅ No missing values found in the dataset!")
        else:
            missing_df = pd.DataFrame({
                'Column': null_counts[null_counts > 0].index,
                'Missing Count': null_counts[null_counts > 0].values,
                'Missing %': (null_counts[null_counts > 0].values / len(st.session_state['cleaned_df']) * 100).round(2)
            })
            
            st.markdown("#### Columns with Missing Values")
            st.dataframe(missing_df, use_container_width=True)
            
            st.markdown("---")
            
            with st.expander("🔧 Apply Missing Value Strategy", expanded=True):
                strategy = st.selectbox(
                    "Select strategy",
                    ["Drop Rows", "Mean (numeric only)", "Median (numeric only)", "Mode", "Constant Fill"],
                    key="missing_strategy"
                )
                
                if strategy != "Drop Rows":
                    cols_with_missing = null_counts[null_counts > 0].index.tolist()
                    selected_column = st.selectbox(
                        "Select column to fill",
                        cols_with_missing,
                        key="missing_column"
                    )
                    
                    if strategy == "Constant Fill":
                        constant_value = st.text_input(
                            "Enter constant value to fill missing cells",
                            key="constant_fill_value"
                        )
                    else:
                        constant_value = None
                else:
                    selected_column = None
                    constant_value = None
                
                strategy_map = {
                    "Drop Rows": "drop",
                    "Mean (numeric only)": "mean",
                    "Median (numeric only)": "median",
                    "Mode": "mode",
                    "Constant Fill": "constant"
                }
                
                if st.button("Apply Strategy", key="apply_missing_btn", use_container_width=True):
                    cleaned, num_affected = fill_missing_values(
                        st.session_state['cleaned_df'],
                        strategy_map[strategy],
                        selected_column,
                        constant_value
                    )
                    
                    if num_affected > 0:
                        st.session_state['cleaned_df'] = cleaned
                        st.success(f"✅ Applied {strategy.lower()}. Affected rows: {num_affected}. New row count: {len(cleaned)}")
                        st.rerun()
                    else:
                        st.warning("No missing values were affected by this operation.")
    
    with cleaning_tabs[2]:
        st.subheader("📊 Detect & Remove Outliers (IQR Method)")
        
        numeric_cols = get_numeric_columns(st.session_state['cleaned_df'])
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for outlier detection.")
        else:
            selected_columns = st.multiselect(
                "Select numeric columns to check for outliers",
                numeric_cols,
                default=numeric_cols[:1] if numeric_cols else [],
                key="outlier_columns"
            )
            
            if selected_columns:
                st.markdown("---")
                
                outlier_stats = []
                for col in selected_columns:
                    outlier_mask, lower, upper = detect_outliers_iqr(st.session_state['cleaned_df'], col)
                    num_outliers = outlier_mask.sum()
                    outlier_stats.append({
                        'Column': col,
                        'Lower Bound': f"{lower:.2f}" if lower is not None else "N/A",
                        'Upper Bound': f"{upper:.2f}" if upper is not None else "N/A",
                        'Outlier Count': num_outliers,
                        'Outlier %': f"{(num_outliers / len(st.session_state['cleaned_df']) * 100):.2f}%"
                    })
                
                st.markdown("#### Outlier Summary")
                st.dataframe(pd.DataFrame(outlier_stats), use_container_width=True)
                
                st.markdown("---")
                
                with st.expander("📈 Visualize Pre/Post Distribution", expanded=True):
                    if len(selected_columns) > 0:
                        for col in selected_columns:
                            try:
                                outlier_mask, lower, upper = detect_outliers_iqr(st.session_state['cleaned_df'], col)
                                df_no_outliers = st.session_state['cleaned_df'][~outlier_mask]
                                
                                fig = go.Figure()
                                
                                fig.add_trace(go.Box(
                                    y=st.session_state['cleaned_df'][col],
                                    name='Before',
                                    marker_color='lightblue'
                                ))
                                
                                fig.add_trace(go.Box(
                                    y=df_no_outliers[col],
                                    name='After',
                                    marker_color='lightgreen'
                                ))
                                
                                fig.update_layout(
                                    title=f"Distribution of {col} - Pre/Post Outlier Removal",
                                    yaxis_title=col,
                                    height=400,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.warning(f"Could not visualize {col}: {str(e)}")
                
                st.markdown("---")
                
                with st.expander("🔧 Apply Outlier Removal", expanded=True):
                    if st.button("Remove Outliers", key="remove_outliers_btn", use_container_width=True):
                        cleaned, num_removed = remove_outliers_iqr(st.session_state['cleaned_df'], selected_columns)
                        st.session_state['cleaned_df'] = cleaned
                        st.success(f"✅ Removed {num_removed} outlier row(s). New row count: {len(cleaned)}")
                        st.rerun()


def render_eda_tab(df: pd.DataFrame):
    """
    Render the complete EDA tab with all analysis sections.
    
    Args:
        df: The pandas DataFrame to analyze
    """
    if df is None or df.empty:
        st.info("👋 Please upload a data file to start exploratory data analysis.")
        return
    
    st.header("🔬 Exploratory Data Analysis")
    
    eda_tabs = st.tabs([
        "📊 Univariate",
        "🔍 Bivariate",
        "🔥 Correlation",
        "🔎 Missing Data"
    ])
    
    with eda_tabs[0]:
        render_univariate_analysis(df)
    
    with eda_tabs[1]:
        render_bivariate_analysis(df)
    
    with eda_tabs[2]:
        render_correlation_heatmap(df)
    
    with eda_tabs[3]:
        render_missing_data_analysis(df)


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Data Analysis App",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    st.title("📊 Data Analysis Application")
    st.markdown("Upload your data file to begin analysis")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a CSV or Excel file to analyze"
        )
        
        if uploaded_file is not None:
            if st.session_state['filename'] != uploaded_file.name:
                with st.spinner('Loading data...'):
                    df, error = read_uploaded_file(uploaded_file)
                    update_session_data(df, uploaded_file.name, error)
                    
                    if error is None:
                        st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                    else:
                        st.error(f"❌ Failed to load file")
        
        if st.session_state['df'] is not None:
            st.markdown("---")
            if st.button("🔄 Reset Data", use_container_width=True):
                reset_session_data()
                st.rerun()
            
            st.markdown("---")
            st.markdown(f"**Current File:** {st.session_state['filename']}")
    
    if st.session_state['upload_error'] is not None:
        st.error(f"⚠️ Upload Error: {st.session_state['upload_error']}")
        st.warning("Please try uploading a different file or check the file format.")
    
    if st.session_state['df'] is not None:
        main_tabs = st.tabs(["📋 Data Preview", "🔬 EDA", "🧹 Data Cleaning"])
        
        with main_tabs[0]:
            render_data_preview(st.session_state['df'])
        
        with main_tabs[1]:
            render_eda_tab(st.session_state['df'])
        
        with main_tabs[2]:
            render_data_cleaning_tab(st.session_state['df'])
    else:
        render_empty_state()


if __name__ == "__main__":
    main()
