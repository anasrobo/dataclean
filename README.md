# Data Analysis Application

A foundational Streamlit application for data analysis with CSV and Excel file upload capabilities.

## Features

- 📤 **File Upload**: Support for CSV and Excel (.xlsx, .xls) files
- 🔄 **Session State Management**: Persistent data storage across user interactions
- 📊 **Data Preview**: View data head, shape, dtypes, and summary statistics
- ⚠️ **Error Handling**: Comprehensive error catching for file parsing issues
- 🎨 **User-Friendly Interface**: Empty state messaging and warning banners
- 🔧 **Utility Helpers**: Functions to reset and update session data
- 📑 **Extensible Layout**: Ready for additional tabs and modules

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Application Structure

### Main Components

- **init_session_state()**: Initializes session state variables for data persistence
- **reset_session_data()**: Resets all session data to initial state
- **update_session_data()**: Updates session with new DataFrame and metadata
- **read_uploaded_file()**: Handles file reading with comprehensive error handling
- **render_data_preview()**: Displays data metrics, head, dtypes, and statistics
- **render_empty_state()**: Shows welcome message when no data is loaded
- **main()**: Application entry point with layout configuration

### Session State Variables

- `df`: The loaded pandas DataFrame
- `filename`: Name of the uploaded file
- `upload_error`: Error message if upload/parsing failed

## Supported File Formats

- CSV (.csv)
- Excel (.xlsx, .xls)

## Data Preview Features

Once data is loaded, you can view:

1. **Head Tab**: First 10 rows of the dataset
2. **Data Types Tab**: Column names, data types, null counts
3. **Summary Statistics Tab**: Statistical summary using pandas describe()

## Error Handling

The application handles various error scenarios:

- Empty files
- Parsing errors
- Unsupported file formats
- General read errors

All errors are displayed with user-friendly warning banners.

## Future Extensions

The application is designed to be extensible. Additional tabs and modules can be easily integrated into the main layout.
