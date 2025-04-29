# Housing Price Prediction Model 🏠📈

This project implements a machine learning pipeline to predict housing prices using linear regression models, based on the Housing.csv dataset. The code performs data preprocessing, feature engineering, model training, evaluation, and visualization, comparing simple and multiple linear regression models.

Run the analysis on : [Colab Notebook](https://colab.research.google.com/drive/1KVXxmDTE7KpyCdUcjVpWxkF0vRkjtJuW?usp=sharing) , [Kaggle Notebook](https://www.kaggle.com/code/shubham1921/housing-price-prediction-model)  

## Project Overview ℹ️
The goal of this project is to predict housing prices (in INR) using features such as area, number of bedrooms, and amenities like air conditioning and furnishing status. The code employs:

- **Simple Linear Regression**: Uses log-transformed area as the sole predictor.
- **Multiple Linear Regression**: Incorporates multiple features, selected via Recursive Feature Elimination (RFE).
- **Robust Preprocessing**: Handles outliers, categorical encoding, multicollinearity, and data scaling.
- **Evaluation Metrics**: Includes MAE, MSE, R², and Adjusted R² on both log and original scales, with cross-validation for the multiple regression model.
- **Visualizations**: Generates plots to analyze feature correlations, model performance, and residuals.

The code is written in Python and uses popular libraries like Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.

## Dataset 📊
The dataset (`Housing.csv`) contains housing data with the following columns:

- `price`: Target variable (housing price in INR).
- `area`: House area in square feet.
- `bedrooms`, `bathrooms`, `stories`, `parking`: Numeric features.
- `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`: Categorical features.

**Note**: The dataset must be in the same directory as the script or the file path in `pd.read_csv('Housing.csv')` should be updated.

## Features 🔍
- **Data Preprocessing**: Outlier removal using IQR, log-transformation of area and price.
- **Feature Engineering**: One-hot encoding of categorical variables, dropping non-numeric columns.
- **Data Integrity**: Checks for NaN/infinite values and index alignment.
- **Feature Analysis**: Correlation matrix heatmap and Variance Inflation Factor (VIF) to address multicollinearity.
- **Modeling**: Simple and multiple linear regression with feature scaling and RFE.
- **Evaluation**: Comprehensive metrics (MAE, MSE, R², Adjusted R²) and 5-fold cross-validation.
- **Visualizations**: Correlation heatmap, scatter plots for predictions, and residual plots.
- **Output**: Processed dataset saved as `processed_housing_data.csv`.

## Folder Structure 📁
```
AI-ML-Internship-Task3/
│
├── data/                       📂 Directory for dataset files
│   └── Housing.csv             📊 Input dataset with housing data
│
├── outputs/                    📂 Directory for generated outputs
│   ├── processed_housing_data.csv  💾 Processed dataset with encoded features
│   ├── correlation_matrix.png      📈 Feature correlation heatmap
│   ├── simple_regression_plot.png  📉 Simple linear regression predictions
│   ├── multiple_regression_plot.png 📉 Multiple linear regression predictions
│   └── residual_plot.png           📉 Residual plot for multiple regression
│
├── housing_prediction.py       🐍 Main Python script with the pipeline
├── README.md                   📝 Project documentation (this file)
└── task 3.pdf                  📜 pdf file (Task)
```

**Notes**:
- The `data/` and `outputs/` directories are assumed for organization. If not present, create them or place `Housing.csv` in the root directory.
- The script generates output files in the root directory by default; modify the script to save to `outputs/` if desired.

## Prerequisites 🛠️
Ensure you have the following installed:
- Python 3.8+ 🐍
- Jupyter Notebook or a Python IDE (e.g., VS Code, PyCharm) 📓
- Libraries:
  - pandas 🐼
  - numpy 🔢
  - matplotlib 📉
  - seaborn 🎨
  - scikit-learn 🤖
  - statsmodels 📊

## Installation 🚀
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd housing-price-prediction
   ```

2. **Install Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
   ```

3. **Prepare the Dataset**:
   - Place `Housing.csv` in the `data/` directory or update the file path in the code.

4. **Set Up Jupyter Notebook** (if using):
   ```bash
   pip install jupyter
   jupyter notebook
   ```

## Usage ▶️
1. Open the script (`housing_prediction.py`) or Jupyter Notebook.
2. Ensure `Housing.csv` is in the correct directory (e.g., `data/Housing.csv`).
3. Run the code sequentially to:
   - Load and preprocess the data.
   - Train and evaluate simple and multiple linear regression models.
   - Generate visualizations and save the processed dataset.

**Example Command** (if saved as a script):
```bash
python housing_prediction.py
```

## Pipeline Steps 🛤️
The code is organized into 9 steps:
1. **Data Loading and Preprocessing**: Loads `Housing.csv`, checks data info, removes outliers (IQR method), and log-transforms area.
2. **Feature Engineering and Data Cleaning**: One-hot encodes categorical variables, separates features and target (`price`), and drops non-numeric columns.
3. **Data Integrity Checks**: Removes NaN/infinite values, resets indices, and verifies index alignment.
4. **Feature Analysis and Multicollinearity Check**: Generates a correlation heatmap and computes VIF, dropping features with VIF > 20.
5. **Data Preparation for Modeling**: Scales features, splits data (80% train, 20% test), and selects features using RFE.
6. **Model Training and Prediction**: Trains simple and multiple linear regression models and makes predictions.
7. **Evaluation Function**: Defines a function to compute MAE, MSE, R², Adjusted R², and cross-validation scores.
8. **Model Evaluation and Visualization**: Evaluates both models and generates scatter plots and residual plots.
9. **Result Interpretation and Data Saving**: Prints model coefficients and saves the processed dataset.

## Outputs 📈
- **Console Outputs**:
  - Dataset info, missing values, and shape.
  - Feature correlations, VIF scores, and selected features.
  - Evaluation metrics for both models (log and original scales).
  - Cross-validation R² scores for multiple regression.
  - Model coefficients.
- **Files**:
  - `processed_housing_data.csv`: Cleaned and processed dataset.
  - Visualization files:
    - `correlation_matrix.png`: Feature correlation heatmap.
    - `simple_regression_plot.png`: Simple regression predictions.
    - `multiple_regression_plot.png`: Multiple regression predictions.
    - `residual_plot.png`: Residuals for multiple regression.

## Visualizations 📊
The code generates the following plots:
- **Correlation Matrix**: Heatmap showing feature correlations.
- **Simple Linear Regression Plot**: Scatter plot of actual vs. predicted prices using log-transformed area.
- **Multiple Linear Regression Plot**: Scatter plot of actual vs. predicted prices with an ideal line.
- **Residual Plot**: Residuals vs. predicted prices, highlighting outliers.

## Contributing 🤝
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

Please ensure your code follows the existing style and includes tests if applicable.

**Happy predicting!** 🎉 If you encounter issues or have questions, feel free to open an issue on the repository.
