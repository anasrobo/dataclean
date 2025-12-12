import streamlit as st
import pandas as pd
from typing import Optional
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE


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
    if 'feature_engineering_log' not in st.session_state:
        st.session_state['feature_engineering_log'] = []
    if 'encoded_columns' not in st.session_state:
        st.session_state['encoded_columns'] = []
    if 'cleaning_log' not in st.session_state:
        st.session_state['cleaning_log'] = []


def reset_session_data():
    """Reset all session data to initial state."""
    st.session_state['df'] = None
    st.session_state['filename'] = None
    st.session_state['upload_error'] = None
    st.session_state['cleaned_df'] = None
    st.session_state['feature_engineering_log'] = []
    st.session_state['encoded_columns'] = []
    st.session_state['cleaning_log'] = []


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


def handle_missing_values(df: pd.DataFrame, strategy: str, columns: list = None) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame using specified strategy.
    
    Args:
        df: The pandas DataFrame
        strategy: Strategy to handle missing values ('drop', 'mean', 'median', 'mode', 'forward_fill', 'backward_fill')
        columns: Specific columns to apply the strategy to (None for all columns)
        
    Returns:
        DataFrame with missing values handled
    """
    df_cleaned = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    
    if strategy == 'drop':
        df_cleaned = df_cleaned.dropna(subset=columns)
    elif strategy == 'mean':
        for col in columns:
            if col in get_numeric_columns(df_cleaned):
                df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
    elif strategy == 'median':
        for col in columns:
            if col in get_numeric_columns(df_cleaned):
                df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    elif strategy == 'mode':
        for col in columns:
            if not df_cleaned[col].mode().empty:
                df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
    elif strategy == 'forward_fill':
        df_cleaned[columns] = df_cleaned[columns].fillna(method='ffill')
    elif strategy == 'backward_fill':
        df_cleaned[columns] = df_cleaned[columns].fillna(method='bfill')
    
    return df_cleaned


def remove_duplicates(df: pd.DataFrame, subset: list = None, keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: The pandas DataFrame
        subset: Columns to consider for identifying duplicates (None for all columns)
        keep: Which duplicates to keep ('first', 'last', False for remove all)
        
    Returns:
        DataFrame with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep=keep)


def render_cleaning_tab(df: pd.DataFrame):
    """
    Render the data cleaning tab with various cleaning operations.
    
    Args:
        df: The pandas DataFrame
    """
    if df is None or df.empty:
        st.info("👋 Please upload a data file to start data cleaning.")
        return
    
    try:
        st.header("🧹 Data Cleaning")
        
        cleaning_tabs = st.tabs([
            "🔧 Handle Missing Values",
            "🔄 Remove Duplicates",
            "📋 Cleaning Summary"
        ])
        
        # Handle Missing Values Tab
        with cleaning_tabs[0]:
            st.subheader("🔧 Handle Missing Values")
            
            null_counts = df.isnull().sum()
            cols_with_nulls = null_counts[null_counts > 0].index.tolist()
            
            if len(cols_with_nulls) == 0:
                st.success("✅ No missing values found in the dataset!")
            else:
                st.warning(f"⚠️ Found missing values in {len(cols_with_nulls)} column(s)")
                
                with st.expander("📊 View Missing Data Details", expanded=True):
                    missing_df = pd.DataFrame({
                        'Column': cols_with_nulls,
                        'Missing Count': [null_counts[col] for col in cols_with_nulls],
                        'Missing %': [(null_counts[col] / len(df)) * 100 for col in cols_with_nulls]
                    })
                    st.dataframe(missing_df, use_container_width=True)
                
                with st.expander("⚙️ Missing Value Options", expanded=True):
                    strategy = st.selectbox(
                        "Select strategy to handle missing values",
                        ["drop", "mean", "median", "mode", "forward_fill", "backward_fill"],
                        help="Choose how to handle missing values"
                    )
                    
                    apply_to_all = st.checkbox("Apply to all columns with missing values", value=True)
                    
                    if not apply_to_all:
                        selected_cols = st.multiselect(
                            "Select columns to apply strategy",
                            cols_with_nulls,
                            default=cols_with_nulls
                        )
                    else:
                        selected_cols = cols_with_nulls
                    
                    if st.button("Apply Strategy", key="apply_missing_strategy"):
                        try:
                            df_cleaned = handle_missing_values(df, strategy, selected_cols)
                            st.session_state['df'] = df_cleaned
                            
                            log_entry = f"✅ Handled missing values using '{strategy}' strategy on {len(selected_cols)} column(s)"
                            st.session_state['cleaning_log'].append(log_entry)
                            
                            st.success(f"✅ Successfully applied {strategy} strategy!")
                            st.info(f"Rows before: {len(df)} → Rows after: {len(df_cleaned)}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error applying strategy: {str(e)}")
        
        # Remove Duplicates Tab
        with cleaning_tabs[1]:
            st.subheader("🔄 Remove Duplicates")
            
            duplicate_count = df.duplicated().sum()
            
            if duplicate_count == 0:
                st.success("✅ No duplicate rows found in the dataset!")
            else:
                st.warning(f"⚠️ Found {duplicate_count} duplicate row(s)")
                st.metric("Duplicate Rows", duplicate_count)
                st.metric("Duplicate Percentage", f"{(duplicate_count / len(df)) * 100:.2f}%")
                
                with st.expander("⚙️ Duplicate Removal Options", expanded=True):
                    keep_option = st.radio(
                        "Which duplicates to keep",
                        ["first", "last", "none"],
                        help="Choose which duplicate to keep"
                    )
                    
                    all_cols = df.columns.tolist()
                    consider_cols = st.multiselect(
                        "Consider only these columns for duplicate detection (leave empty for all)",
                        all_cols
                    )
                    
                    if st.button("Remove Duplicates", key="remove_duplicates_btn"):
                        try:
                            keep_val = False if keep_option == "none" else keep_option
                            subset = consider_cols if consider_cols else None
                            
                            df_cleaned = remove_duplicates(df, subset=subset, keep=keep_val)
                            st.session_state['df'] = df_cleaned
                            
                            removed_count = len(df) - len(df_cleaned)
                            log_entry = f"✅ Removed {removed_count} duplicate row(s)"
                            st.session_state['cleaning_log'].append(log_entry)
                            
                            st.success(f"✅ Successfully removed {removed_count} duplicate(s)!")
                            st.info(f"Rows before: {len(df)} → Rows after: {len(df_cleaned)}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error removing duplicates: {str(e)}")
        
        # Cleaning Summary Tab
        with cleaning_tabs[2]:
            st.subheader("📋 Cleaning Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Rows", df.shape[0])
            with col2:
                st.metric("Current Columns", df.shape[1])
            with col3:
                st.metric("Cleaning Operations", len(st.session_state.get('cleaning_log', [])))
            
            if st.session_state.get('cleaning_log'):
                st.markdown("#### Cleaning Operation Log")
                for log in st.session_state['cleaning_log']:
                    st.write(log)
            else:
                st.info("No cleaning operations performed yet.")
            
            st.markdown("#### Current Data Quality")
            col1, col2 = st.columns(2)
            
            with col1:
                missing_count = df.isnull().sum().sum()
                st.metric("Total Missing Values", int(missing_count))
            
            with col2:
                duplicate_count = df.duplicated().sum()
                st.metric("Duplicate Rows", int(duplicate_count))
    
    except Exception as e:
        st.error(f"❌ Error in data cleaning tab: {str(e)}")
        st.warning("Please try refreshing the page or uploading a new file.")


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
def apply_one_hot_encoding(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, list]:
    """
    Apply one-hot encoding to specified categorical columns.
    
    Args:
        df: The pandas DataFrame
        columns: List of column names to encode
        
    Returns:
        Tuple of (encoded DataFrame, list of new column names)
    """
    new_columns = []
    df_encoded = df.copy()
    
    for col in columns:
        if col not in df_encoded.columns:
            continue
            
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_array = encoder.fit_transform(df_encoded[[col]])
            encoded_feature_names = encoder.get_feature_names_out([col])
            
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=df_encoded.index)
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
            
            new_columns.extend(encoded_feature_names.tolist())
        except Exception as e:
            st.error(f"Failed to apply one-hot encoding to {col}: {str(e)}")
            continue
    
    return df_encoded, new_columns


def apply_label_encoding(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, list]:
    """
    Apply label encoding to specified categorical columns.
    
    Args:
        df: The pandas DataFrame
        columns: List of column names to encode
        
    Returns:
        Tuple of (encoded DataFrame, list of encoded column names)
    """
    new_columns = []
    df_encoded = df.copy()
    
    for col in columns:
        if col not in df_encoded.columns:
            continue
            
        try:
            encoder = LabelEncoder()
            df_encoded[f"{col}_encoded"] = encoder.fit_transform(df_encoded[col].astype(str))
            new_columns.append(f"{col}_encoded")
        except Exception as e:
            st.error(f"Failed to apply label encoding to {col}: {str(e)}")
            continue
    
    return df_encoded, new_columns


def render_categorical_encoding_tab(df: pd.DataFrame):
    """
    Render categorical encoding interface with one-hot and label encoding options.
    
    Args:
        df: The pandas DataFrame
    """
    st.subheader("🏷️ Categorical Encoding")
    
    categorical_cols = get_categorical_columns(df)
    
    if len(categorical_cols) == 0:
        st.info("No categorical columns found in the dataset.")
        return
    
    st.markdown("Transform categorical variables into numerical format for machine learning models.")
    
    with st.expander("⚙️ Encoding Configuration", expanded=True):
        encoding_method = st.radio(
            "Select encoding method",
            ["One-Hot Encoding", "Label Encoding"],
            horizontal=True,
            help="One-Hot creates binary columns, Label assigns numeric values"
        )
        
        st.markdown(f"**Available Categorical Columns:** {len(categorical_cols)}")
        
        selected_cols = st.multiselect(
            "Select categorical columns to encode",
            categorical_cols,
            help="Choose one or more columns to apply the selected encoding"
        )
        
        if selected_cols:
            st.info(f"Selected {len(selected_cols)} column(s) for {encoding_method}")
    
    if selected_cols and st.button("Apply Encoding", key="apply_encoding_btn", type="primary"):
        try:
            df_encoded = df.copy()
            new_columns = []
            
            if encoding_method == "One-Hot Encoding":
                df_encoded, new_columns = apply_one_hot_encoding(df_encoded, selected_cols)
            else:
                df_encoded, new_columns = apply_label_encoding(df_encoded, selected_cols)
            
            st.session_state['df'] = df_encoded
            st.session_state['encoded_columns'].extend(new_columns)
            
            log_entry = f"✅ Applied {encoding_method} to: {', '.join(selected_cols)}"
            st.session_state['feature_engineering_log'].append(log_entry)
            
            st.success(f"✅ Successfully applied {encoding_method}!")
            st.info(f"New columns created: {', '.join(new_columns)}")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error applying encoding: {str(e)}")
    
    if st.session_state.get('encoded_columns'):
        st.markdown("---")
        with st.expander("📋 Previously Encoded Columns", expanded=False):
            for col in st.session_state['encoded_columns']:
                st.write(f"• {col}")


def render_smote_balancing_tab(df: pd.DataFrame):
    """
    Render SMOTE balancing interface.
    
    Args:
        df: The pandas DataFrame
    """
    st.subheader("⚖️ SMOTE Balancing")
    
    all_cols = df.columns.tolist()
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for SMOTE balancing.")
        return
    
    st.markdown("Balance imbalanced datasets using Synthetic Minority Over-sampling Technique (SMOTE).")
    
    with st.expander("⚙️ SMOTE Configuration", expanded=True):
        target_col = st.selectbox(
            "Select target column for balancing",
            all_cols,
            help="Choose the column containing the target classes to balance"
        )
        
        st.info("SMOTE will oversample the minority class to balance the distribution.")
    
    if target_col:
        class_counts = df[target_col].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Class Distribution (Before)")
            fig_before = px.bar(
                x=class_counts.index.astype(str),
                y=class_counts.values,
                labels={'x': target_col, 'y': 'Count'},
                title="Before SMOTE"
            )
            fig_before.update_layout(height=400)
            st.plotly_chart(fig_before, use_container_width=True)
            
            for idx, count in class_counts.items():
                st.write(f"{idx}: {count} samples")
        
        with col2:
            if st.button("Apply SMOTE Balancing", key="apply_smote_btn"):
                try:
                    if target_col in categorical_cols:
                        st.error(f"❌ Target column '{target_col}' is categorical. SMOTE requires numeric or encoded target.")
                    elif target_col in numeric_cols:
                        numeric_features = df[numeric_cols].copy()
                        
                        nan_mask = numeric_features.isnull().any(axis=1)
                        if nan_mask.any():
                            st.warning(f"⚠️ Removing {nan_mask.sum()} rows with missing values in numeric features.")
                            numeric_features = numeric_features[~nan_mask]
                            target_series = df.loc[numeric_features.index, target_col]
                        else:
                            target_series = df[target_col]
                        
                        smote = SMOTE(random_state=42)
                        X_resampled, y_resampled = smote.fit_resample(numeric_features, target_series)
                        
                        df_resampled = pd.DataFrame(X_resampled, columns=numeric_features.columns)
                        df_resampled[target_col] = y_resampled
                        
                        st.session_state['df'] = df_resampled
                        log_entry = f"✅ Applied SMOTE balancing with target: {target_col}"
                        st.session_state['feature_engineering_log'].append(log_entry)
                        
                        st.success("✅ SMOTE balancing applied successfully!")
                        
                        new_class_counts = pd.Series(y_resampled).value_counts()
                        st.markdown("#### Class Distribution (After)")
                        fig_after = px.bar(
                            x=new_class_counts.index.astype(str),
                            y=new_class_counts.values,
                            labels={'x': target_col, 'y': 'Count'},
                            title="After SMOTE"
                        )
                        fig_after.update_layout(height=400)
                        st.plotly_chart(fig_after, use_container_width=True)
                        
                        for idx, count in new_class_counts.items():
                            st.write(f"{idx}: {count} samples")
                        
                        st.info(f"Original size: {len(df)} → New size: {len(df_resampled)}")
                        st.rerun()
                    else:
                        st.error("❌ Unable to apply SMOTE to the selected target column.")
                except Exception as e:
                    st.error(f"❌ Error applying SMOTE: {str(e)}")


def render_feature_engineering_summary(df: pd.DataFrame):
    """
    Render summary of feature engineering operations.
    
    Args:
        df: The pandas DataFrame
    """
    st.subheader("📋 Feature Engineering Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Columns", df.shape[1])
    with col2:
        st.metric("Total Rows", df.shape[0])
    with col3:
        st.metric("Operations Performed", len(st.session_state.get('feature_engineering_log', [])))
    
    if st.session_state.get('feature_engineering_log'):
        st.markdown("#### Operation Log")
        for log in st.session_state['feature_engineering_log']:
            st.write(log)
    
    st.markdown("#### Current Dataset Info")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Numeric Columns:** {len(get_numeric_columns(df))}")
        for col in get_numeric_columns(df):
            st.caption(col)
    
    with col2:
        st.write(f"**Categorical Columns:** {len(get_categorical_columns(df))}")
        for col in get_categorical_columns(df):
            st.caption(col)


def render_feature_engineering_tab(df: pd.DataFrame):
    """
    Render the complete feature engineering tab.
    
    Args:
        df: The pandas DataFrame
    """
    if df is None or df.empty:
        st.info("👋 Please upload a data file to start feature engineering.")
        return
    
    try:
        st.header("⚡ Feature Engineering & Preprocessing")
        
        fe_tabs = st.tabs([
            "🏷️ Categorical Encoding",
            "⚖️ SMOTE Balancing",
            "📋 Summary"
        ])
        
        with fe_tabs[0]:
            render_categorical_encoding_tab(df)
        
        with fe_tabs[1]:
            render_smote_balancing_tab(df)
        
        with fe_tabs[2]:
            render_feature_engineering_summary(df)
    
    except Exception as e:
        st.error(f"❌ Error in feature engineering tab: {str(e)}")
        st.warning("Please try refreshing the page.")


def render_eda_tab(df: pd.DataFrame):
    """
    Render the complete EDA tab with all analysis sections.
    
    Args:
        df: The pandas DataFrame to analyze
    """
    if df is None or df.empty:
        st.info("👋 Please upload a data file to start exploratory data analysis.")
        return
    
    try:
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
    
    except Exception as e:
        st.error(f"❌ Error in EDA tab: {str(e)}")
        st.warning("Please try refreshing the page or check your data.")


def render_export_tab(df: pd.DataFrame):
    """
    Render the export tab with download functionality.
    
    Args:
        df: The pandas DataFrame to export
    """
    if df is None or df.empty:
        st.info("👋 Please upload and process data before exporting.")
        return
    
    try:
        st.header("📥 Export Data")
        
        st.markdown("""
        Export your processed data to CSV format. All transformations and cleaning operations 
        will be included in the exported file.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows to Export", df.shape[0])
        with col2:
            st.metric("Columns to Export", df.shape[1])
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024
            st.metric("File Size (approx)", f"{memory_usage:.2f} KB")
        
        st.markdown("---")
        
        with st.expander("📊 Preview Data to Export", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)
        
        with st.expander("⚙️ Export Options", expanded=True):
            export_filename = st.text_input(
                "Filename (without extension)",
                value=st.session_state.get('filename', 'processed_data').replace('.csv', '').replace('.xlsx', '').replace('.xls', ''),
                help="Enter the desired filename for the export"
            )
            
            include_index = st.checkbox("Include row index", value=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_separator = st.selectbox(
                    "CSV Separator",
                    [",", ";", "\t", "|"],
                    index=0,
                    help="Choose the delimiter for CSV export"
                )
            
            with col2:
                encoding = st.selectbox(
                    "Encoding",
                    ["utf-8", "utf-8-sig", "latin1", "iso-8859-1"],
                    index=0,
                    help="Choose the character encoding"
                )
        
        st.markdown("---")
        
        try:
            csv_data = df.to_csv(index=include_index, sep=csv_separator, encoding=encoding)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name=f"{export_filename}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
        
        except Exception as e:
            st.error(f"❌ Error preparing file for download: {str(e)}")
        
        st.markdown("---")
        
        st.markdown("#### 📋 Export Summary")
        
        if st.session_state.get('cleaning_log') or st.session_state.get('feature_engineering_log'):
            st.markdown("##### Operations Applied:")
            
            all_logs = []
            if st.session_state.get('cleaning_log'):
                all_logs.extend(st.session_state['cleaning_log'])
            if st.session_state.get('feature_engineering_log'):
                all_logs.extend(st.session_state['feature_engineering_log'])
            
            for i, log in enumerate(all_logs, 1):
                st.write(f"{i}. {log}")
        else:
            st.info("No transformations applied to this dataset.")
        
        st.markdown("##### Column Information:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Numeric Columns:** {len(get_numeric_columns(df))}")
        with col2:
            st.write(f"**Categorical Columns:** {len(get_categorical_columns(df))}")
    
    except Exception as e:
        st.error(f"❌ Error in export tab: {str(e)}")
        st.warning("Please try refreshing the page.")


def main():
    """Main application entry point."""
    try:
        st.set_page_config(
            page_title="Data Analysis App",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        pass
    
    try:
        init_session_state()
        
        st.title("📊 Data Analysis Application")
        st.markdown("Upload your data file to begin comprehensive analysis and transformation")
        
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            with st.expander("📤 Upload Data", expanded=True):
                uploaded_file = st.file_uploader(
                    "Choose your data file",
                    type=['csv', 'xlsx', 'xls'],
                    help="Upload a CSV or Excel file to analyze"
                )
                
                if uploaded_file is not None:
                    if st.session_state['filename'] != uploaded_file.name:
                        with st.spinner('Loading data...'):
                            try:
                                df, error = read_uploaded_file(uploaded_file)
                                update_session_data(df, uploaded_file.name, error)
                                
                                if error is None:
                                    st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                                else:
                                    st.error(f"❌ Failed to load file")
                            except Exception as e:
                                st.error(f"❌ Unexpected error: {str(e)}")
            
            if st.session_state['df'] is not None:
                st.markdown("---")
                
                with st.expander("📊 Dataset Info", expanded=False):
                    st.metric("Current File", st.session_state['filename'])
                    st.metric("Rows", st.session_state['df'].shape[0])
                    st.metric("Columns", st.session_state['df'].shape[1])
                    
                    total_operations = (
                        len(st.session_state.get('cleaning_log', [])) +
                        len(st.session_state.get('feature_engineering_log', []))
                    )
                    st.metric("Total Operations", total_operations)
                
                st.markdown("---")
                
                if st.button("🔄 Reset All Data", use_container_width=True, type="secondary"):
                    reset_session_data()
                    st.rerun()
        
        if st.session_state['upload_error'] is not None:
            st.error(f"⚠️ Upload Error: {st.session_state['upload_error']}")
            st.warning("Please try uploading a different file or check the file format.")
        
        if st.session_state['df'] is not None:
            main_tabs = st.tabs([
                "📋 Upload & Overview",
                "🔬 EDA",
                "🧹 Cleaning",
                "⚡ Feature Engineering",
                "📥 Export"
            ])
            
            with main_tabs[0]:
                try:
                    render_data_preview(st.session_state['df'])
                except Exception as e:
                    st.error(f"❌ Error displaying data preview: {str(e)}")
                    st.warning("Please try refreshing the page.")
            
            with main_tabs[1]:
                try:
                    render_eda_tab(st.session_state['df'])
                except Exception as e:
                    st.error(f"❌ Error in EDA tab: {str(e)}")
                    st.warning("Please try refreshing the page or check your data.")
            
            with main_tabs[2]:
                try:
                    render_cleaning_tab(st.session_state['df'])
                except Exception as e:
                    st.error(f"❌ Error in cleaning tab: {str(e)}")
                    st.warning("Please try refreshing the page.")
            
            with main_tabs[3]:
                try:
                    render_feature_engineering_tab(st.session_state['df'])
                except Exception as e:
                    st.error(f"❌ Error in feature engineering tab: {str(e)}")
                    st.warning("Please try refreshing the page.")
            
            with main_tabs[4]:
                try:
                    render_export_tab(st.session_state['df'])
                except Exception as e:
                    st.error(f"❌ Error in export tab: {str(e)}")
                    st.warning("Please try refreshing the page.")
        else:
            render_empty_state()
    
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
    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        st.warning("The application encountered an unexpected error. Please refresh the page and try again.")
        if st.button("🔄 Refresh Application"):
            st.rerun()


if __name__ == "__main__":
    main()
