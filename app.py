import streamlit as st
import pandas as pd
from typing import Optional
import io


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
        render_data_preview(st.session_state['df'])
    else:
        render_empty_state()
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            Ready for additional tabs and modules
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
