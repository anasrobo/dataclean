# Data Analysis Application

A comprehensive Streamlit application for end-to-end data analysis, from file upload through cleaning, EDA, feature engineering, to data export.

## Features

- 📤 **File Upload**: Support for CSV and Excel (.xlsx, .xls) files with validation
- 🔄 **Session State Management**: Persistent data storage across user interactions and tabs
- 📊 **Data Preview**: View data head, shape, dtypes, and comprehensive summary statistics
- 🧹 **Data Cleaning**: Interactive data cleaning tools
  - 🔧 **Handle Missing Values**: Multiple strategies (drop, mean, median, mode, forward fill, backward fill)
  - 🔄 **Remove Duplicates**: Flexible duplicate removal with custom subset selection
  - 📋 **Cleaning Summary**: Track all cleaning operations with detailed logs
- 🔬 **Exploratory Data Analysis (EDA)**: Interactive visualizations and statistical analysis
  - 📊 **Univariate Analysis**: Histograms for numeric columns, bar charts for categorical columns
  - 🔍 **Bivariate Analysis**: Scatter plots and box plots with customizable axes
  - 🔥 **Correlation Heatmap**: Visual correlation matrix for numeric columns
  - 🔎 **Missing Data Visualization**: Bar charts and statistics for missing values
- ⚡ **Feature Engineering**: Transform and prepare data for machine learning
  - 🏷️ **Categorical Encoding**: One-Hot and Label encoding for categorical variables
  - ⚖️ **SMOTE Balancing**: Balance imbalanced datasets using SMOTE
  - 📋 **Operation Tracking**: Comprehensive logs of all transformations
- 📥 **Export Functionality**: Download processed data as CSV with customizable options
  - Configurable separators and encoding
  - Export summary with all applied operations
  - Preview before download
- ⚠️ **Global Error Handling**: Comprehensive error catching to prevent crashes
- 🎨 **Polished UI**: Expanders for complex controls, success/warning messaging, intuitive navigation
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
- `feature_engineering_log`: List of feature engineering operations performed
- `encoded_columns`: List of newly created encoded columns
- `cleaning_log`: List of data cleaning operations performed

## Supported File Formats

- CSV (.csv)
- Excel (.xlsx, .xls)

## Application Tabs

### 📋 Upload & Overview Tab

Once data is loaded, you can view:

1. **Head**: First 10 rows of the dataset
2. **Data Types**: Column names, data types, null counts
3. **Summary Statistics**: Statistical summary using pandas describe()
4. **Metrics**: Rows, columns, and memory usage

### 🧹 Cleaning Tab

The Cleaning tab provides tools for data quality improvement:

#### 🔧 Handle Missing Values
- View missing data details with counts and percentages
- Apply strategies: drop, mean, median, mode, forward fill, backward fill
- Select specific columns or apply to all columns with missing values
- Track before/after row counts

#### 🔄 Remove Duplicates
- Detect duplicate rows with counts and percentages
- Choose which duplicates to keep (first, last, or none)
- Optionally consider only specific columns for duplicate detection
- View removal statistics

#### 📋 Cleaning Summary
- Current data quality metrics
- Complete log of all cleaning operations performed
- Missing values and duplicate counts

### 🔬 EDA Tab

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

### ⚡ Feature Engineering Tab

The Feature Engineering tab provides tools for data transformation:

#### 🏷️ Categorical Encoding
- **One-Hot Encoding**: Create binary columns for each category
- **Label Encoding**: Assign numeric values to categories
- Select multiple columns to encode at once
- Track newly created encoded columns
- Expandable configuration panel for complex controls

#### ⚖️ SMOTE Balancing
- Balance imbalanced datasets using Synthetic Minority Over-sampling Technique
- Visual before/after class distribution charts
- Automatic handling of missing values
- Works with numeric target columns
- View original vs. resampled dataset sizes

#### 📋 Summary
- View all feature engineering operations performed
- Current dataset information (rows, columns)
- List of numeric and categorical columns
- Operation log for traceability

### 📥 Export Tab

The Export tab allows you to download your processed data:

- **Preview**: View a sample of the data to be exported
- **Export Options**:
  - Custom filename
  - CSV separator selection (comma, semicolon, tab, pipe)
  - Character encoding options (utf-8, utf-8-sig, latin1, iso-8859-1)
  - Include/exclude row index
- **Download Button**: One-click CSV download
- **Export Summary**: View all operations applied to the data
- **Column Information**: Summary of numeric and categorical columns

## Error Handling

The application implements comprehensive global error handling:

### File Upload Errors
- Empty files
- Parsing errors
- Unsupported file formats
- General read errors

### Data Processing Errors
- Invalid column selections
- Missing data in calculations
- Insufficient numeric columns for operations
- Empty or None DataFrames
- Encoding failures
- SMOTE balancing errors

### UI Error Handling
- Try-catch blocks around all major sections
- User-friendly error messages
- Warning banners with actionable guidance
- Graceful degradation when data is unavailable
- Critical error recovery with refresh option

All errors are caught and displayed with user-friendly messages, preventing application crashes.

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

## Dependencies

All required dependencies are listed in `requirements.txt`:

- `streamlit>=1.28.0` - Web application framework
- `pandas>=2.0.0` - Data manipulation and analysis
- `numpy>=1.24.0` - Numerical computing
- `openpyxl>=3.1.0` - Excel file support
- `pytest>=7.0.0` - Testing framework
- `plotly>=5.0.0` - Interactive visualizations
- `scikit-learn>=1.5.0` - Machine learning and preprocessing
- `imbalanced-learn>=0.12.0` - SMOTE and other resampling techniques

## Future Extensions

The application is designed to be extensible. Additional tabs and modules can be easily integrated into the main layout. Potential additions:
- Advanced statistical tests
- Machine learning model training and evaluation
- Time series analysis
- Export functionality for visualizations
- Data profiling reports
- Custom transformation pipelines
