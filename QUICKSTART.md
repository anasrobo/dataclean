# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`.

## Using the Application

1. **Upload a File**
   - Click the "Browse files" button in the sidebar
   - Select a CSV or Excel file (.csv, .xlsx, .xls)
   - The file will be automatically loaded and processed

2. **View Data Preview**
   - Once loaded, you'll see metrics (rows, columns, memory usage)
   - Explore three tabs:
     - **Head**: First 10 rows of your data
     - **Data Types**: Column information including types and null counts
     - **Summary Statistics**: Statistical overview of your data

3. **Reset Data**
   - Click the "🔄 Reset Data" button in the sidebar to clear current data
   - Upload a new file to analyze

## Testing with Sample Data

A sample CSV file (`sample_data.csv`) is included in the repository for testing purposes.

## Error Handling

If you encounter an error during file upload:
- Check that the file format is supported (CSV or Excel)
- Ensure the file is not empty
- Verify the file is not corrupted
- Check the error message displayed in the application

## Running Tests

```bash
# Run unit tests
pytest test_app.py -v
```

## Features Overview

- ✅ CSV and Excel file support
- ✅ Session state management
- ✅ Comprehensive error handling
- ✅ Data preview with multiple views
- ✅ Empty state messaging
- ✅ Reset functionality
- ✅ Extensible architecture for future modules
