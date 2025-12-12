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


def reset_session_data():
    """Reset all session data to initial state."""
    st.session_state['df'] = None
    st.session_state['filename'] = None
    st.session_state['upload_error'] = None


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
        main_tabs = st.tabs(["📋 Data Preview", "🔬 EDA"])
        
        with main_tabs[0]:
            render_data_preview(st.session_state['df'])
        
        with main_tabs[1]:
            render_eda_tab(st.session_state['df'])
    else:
        render_empty_state()


if __name__ == "__main__":
    main()
