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

2. **View Data Preview Tab**
   - Once loaded, you'll see metrics (rows, columns, memory usage)
   - Explore three sub-tabs:
     - **Head**: First 10 rows of your data
     - **Data Types**: Column information including types and null counts
     - **Summary Statistics**: Statistical overview of your data

3. **Explore EDA Tab**
   - Switch to the EDA tab for interactive visualizations
   - **Univariate Analysis**: Select numeric or categorical columns to view distributions
   - **Bivariate Analysis**: Choose X/Y axes and plot type (scatter or box plot)
   - **Correlation Heatmap**: View relationships between numeric columns
   - **Missing Data**: Analyze patterns of missing values in your dataset

4. **Reset Data**
   - Click the "🔄 Reset Data" button in the sidebar to clear current data
   - Upload a new file to analyze

## Testing with Sample Data

Two sample CSV files are included in the repository for testing purposes:
- `sample_data.csv`: Clean dataset with no missing values
- `sample_data_with_missing.csv`: Dataset with missing values to test EDA features

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
- ✅ **Interactive EDA visualizations with Plotly**
  - Univariate analysis (histograms, bar charts)
  - Bivariate analysis (scatter plots, box plots)
  - Correlation heatmaps
  - Missing data visualization
- ✅ Empty state messaging
- ✅ Reset functionality
- ✅ Extensible architecture for future modules
