# Data Analysis Application

A comprehensive Streamlit application for data analysis with CSV and Excel file upload capabilities and exploratory data analysis (EDA) tools.

## Features

- 📤 **File Upload**: Support for CSV and Excel (.xlsx, .xls) files
- 🔄 **Session State Management**: Persistent data storage across user interactions
- 📊 **Data Preview**: View data head, shape, dtypes, and summary statistics
- 🔬 **Exploratory Data Analysis (EDA)**: Interactive visualizations and statistical analysis
  - 📊 **Univariate Analysis**: Histograms for numeric columns, bar charts for categorical columns
  - 🔍 **Bivariate Analysis**: Scatter plots and box plots with customizable axes
  - 🔥 **Correlation Heatmap**: Visual correlation matrix for numeric columns
  - 🔎 **Missing Data Visualization**: Bar charts and statistics for missing values
- ⚠️ **Error Handling**: Comprehensive error catching for file parsing issues and invalid selections
- 🎨 **User-Friendly Interface**: Empty state messaging, warning banners, and graceful degradation
- 🔧 **Utility Helpers**: Functions to reset and update session data
- 📑 **Extensible Layout**: Modular design for easy addition of new features

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

#### Core Functions
- **init_session_state()**: Initializes session state variables for data persistence
- **reset_session_data()**: Resets all session data to initial state
- **update_session_data()**: Updates session with new DataFrame and metadata
- **read_uploaded_file()**: Handles file reading with comprehensive error handling
- **render_data_preview()**: Displays data metrics, head, dtypes, and statistics
- **render_empty_state()**: Shows welcome message when no data is loaded
- **main()**: Application entry point with layout configuration

#### EDA Functions
- **get_numeric_columns()**: Identifies numeric columns in the DataFrame
- **get_categorical_columns()**: Identifies categorical columns in the DataFrame
- **render_univariate_analysis()**: Creates histograms and bar charts for individual columns
- **render_bivariate_analysis()**: Creates scatter and box plots for column relationships
- **render_correlation_heatmap()**: Generates correlation matrix heatmap with Plotly
- **render_missing_data_analysis()**: Visualizes missing data patterns
- **render_eda_tab()**: Main EDA tab orchestrator

### Session State Variables

- `df`: The loaded pandas DataFrame
- `filename`: Name of the uploaded file
- `upload_error`: Error message if upload/parsing failed

## Supported File Formats

- CSV (.csv)
- Excel (.xlsx, .xls)

## Application Tabs

### Data Preview Tab

Once data is loaded, you can view:

1. **Head**: First 10 rows of the dataset
2. **Data Types**: Column names, data types, null counts
3. **Summary Statistics**: Statistical summary using pandas describe()

### EDA Tab

The EDA tab provides comprehensive exploratory data analysis:

#### 📊 Univariate Analysis
- **Numeric Columns**: Histograms with marginal box plots, showing mean, median, and standard deviation
- **Categorical Columns**: Bar charts showing value distributions and unique value counts
- User-selectable columns with automatic plot type detection
- Handles edge cases like no data or invalid selections gracefully

#### 🔍 Bivariate Analysis
- **Scatter Plots**: Explore relationships between two numeric columns
  - Optional color coding by a third variable
  - Automatic trendline fitting for numeric data
  - Correlation coefficient display
- **Box Plots**: Compare distributions across categories
- User-selectable X and Y axes with plot type toggle

#### 🔥 Correlation Heatmap
- Visual correlation matrix for all numeric columns
- Color-coded cells with correlation values
- Table of strong correlations (|r| > 0.5)
- Automatic sizing based on number of columns

#### 🔎 Missing Data Analysis
- Bar charts showing missing value counts and percentages
- Summary table of columns with missing data
- Overall statistics: total missing values, affected columns, missing percentage
- Success message when no missing data is found

## Error Handling

The application handles various error scenarios:

### File Upload Errors
- Empty files
- Parsing errors
- Unsupported file formats
- General read errors

### EDA Errors
- Invalid column selections
- Missing data in calculations
- Insufficient numeric columns for correlation
- Empty or None DataFrames

All errors are displayed with user-friendly warning messages, and the UI gracefully degrades when data is unavailable.

## Testing

Run the test suite:

```bash
pytest test_app.py -v
```

The test suite includes:
- File reading tests
- Session state management tests
- Column type detection tests
- Missing data detection tests
- Correlation computation tests

## Future Extensions

The application is designed to be extensible. Additional tabs and modules can be easily integrated into the main layout. Potential additions:
- Advanced statistical tests
- Machine learning models
- Data transformation tools
- Export functionality for visualizations
