%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 1: Data Loading and Preprocessing
# Load the dataset and print initial information like columns and data types
data = pd.read_csv('Housing.csv')

# Display dataset columns and information
print("Dataset Columns:", data.columns.tolist())
print("\nDataset Info:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Identify categorical columns manually
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Check if categorical columns match object data types in the dataset
object_cols = data.select_dtypes(include=['object']).columns.tolist()
if set(categorical_cols) != set(object_cols):
    print(f"\nWarning: Object columns {object_cols} differ from categorical_cols {categorical_cols}")
    categorical_cols = object_cols

# Remove outliers for 'price' and 'area' based on IQR (Interquartile Range)
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

data = remove_outliers(data, 'price')
data = remove_outliers(data, 'area')
print(f"\nRows after outlier removal: {len(data)}")

# Log-transform 'area' column to normalize its distribution
data['log_area'] = np.log(data['area'])
data = data.drop(columns=['area'])

# Step 2: Feature Engineering and Data Cleaning
# Use one-hot encoding to convert categorical columns into numeric format
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True, dtype=int)
print("\nData Types After Encoding:")
print(data.dtypes)

# Separate features and target variable ('price')
feature_cols = [col for col in data.columns if col != 'price']
X = data[feature_cols]
y = np.log(data['price'])  # Apply log-transformation to target variable

# Check for and drop any non-numeric columns (if any)
non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
if non_numeric_cols:
    print(f"\nDropping non-numeric columns: {non_numeric_cols}")
    X = X.drop(columns=non_numeric_cols)
    feature_cols = [col for col in X.columns]

# Step 3: Data Integrity Checks
# Ensure that there are no NaN or infinite values in the dataset
if X.isna().any().any() or not np.all(np.isfinite(X)):
    print("\nNaN or infinite values found in X:")
    print(X.isna().sum())
    print(np.isinf(X).sum())
    X = X.dropna()  # Drop rows with NaN values
    y = y[X.index]  # Align target variable y with cleaned X
    print("\nDropped rows with NaN/infinite values.")

# Reset index for X and y
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
print(f"\nAfter cleaning: X shape: {X.shape}, y shape: {y.shape}")
print("X index:", X.index.tolist()[:10], "...")
print("y index:", y.index.tolist()[:10], "...")

# Verify index alignment
if not X.index.equals(y.index):
    raise ValueError("X and y indices are not aligned!")

# Step 4: Feature Analysis and Multicollinearity Check
# Visualize correlation between features using a heatmap
print("\nFeature Correlations:")
corr_matrix = X.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()

# Check for multicollinearity among features
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVIF Scores:")
print(vif_data)

# Drop features with VIF greater than 20 to avoid multicollinearity
high_vif = vif_data[vif_data['VIF'] > 20]['Feature'].tolist()
if high_vif:
    print(f"Dropping high VIF features: {high_vif}")
    X = X.drop(columns=high_vif)
    feature_cols = [col for col in X.columns]

# Step 5: Data Preparation for Modeling
# Scale the features to standardize them using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Use RFE with a Linear Regression model to select the most important features
lr = LinearRegression()
n_features = min(10, X.shape[1])  # Dynamically select number of features
rfe = RFE(estimator=lr, n_features_to_select=n_features)
rfe.fit(X_train, y_train)
print("\nSelected Features using RFE:")
selected_features = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
print(selected_features)

# Update X sets to use only the selected features
X_train = X_train[:, rfe.support_]
X_test = X_test[:, rfe.support_]

# Step 6: Model Training and Prediction
# Train both a Simple Linear Regression model and a Multiple Linear Regression model
X_simple = scaler.fit_transform(data[['log_area']])
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y, test_size=0.2, random_state=42)
simple_lr = LinearRegression()
simple_lr.fit(X_train_simple, y_train_simple)

# Train the Multiple Linear Regression model
multiple_lr = LinearRegression()
multiple_lr.fit(X_train, y_train)

# Make predictions for both models
y_pred_simple = simple_lr.predict(X_test_simple)
y_pred_multiple = multiple_lr.predict(X_test)

# Apply inverse log-transformation to the predicted values
y_test_orig = np.exp(y_test)
y_pred_simple_orig = np.exp(y_pred_simple)
y_pred_multiple_orig = np.exp(y_pred_multiple)

# Step 7: Evaluation Function
# Define a function to evaluate model performance (MAE, MSE, R², Adjusted R²)
def evaluate_model(y_true, y_pred, y_true_orig, y_pred_orig, model_name, X_test_model=None):
    mae_log = mean_absolute_error(y_true, y_pred)
    mse_log = mean_squared_error(y_true, y_pred)
    r2_log = r2_score(y_true, y_pred)

    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
    mse_orig = mean_squared_error(y_true_orig, y_pred_orig)
    r2_orig = r2_score(y_true_orig, y_pred_orig)
    
    # Adjusted R² Calculation
    n = X_test_model.shape[0] if X_test_model is not None else len(y_true)
    p = X_test_model.shape[1] if X_test_model is not None else 1
    adjusted_r2 = 1 - (1 - r2_log) * (n - 1) / (n - p - 1)

    # Print evaluation metrics
    print(f"\n{model_name} Evaluation (Log Scale):")
    print(f"MAE: {mae_log:.4f}")
    print(f"MSE: {mse_log:.4f}")
    print(f"R²: {r2_log:.4f}")
    print(f"Adjusted R²: {adjusted_r2:.4f}")

    print(f"\n{model_name} Evaluation (Original Scale):")
    print(f"MAE: {mae_orig:.2f}")
    print(f"MSE: {mse_orig:.2f}")
    print(f"R²: {r2_orig:.4f}")

    # Cross-validation for Multiple Linear Regression
    if model_name == "Multiple Linear Regression":
        y_bins = pd.qcut(y, q=5, labels=False)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y_bins)):
            X_tr, X_val = X_scaled[train_idx][:, rfe.support_], X_scaled[val_idx][:, rfe.support_]
            y_tr, y_val = y.values[train_idx], y.values[val_idx]
            model = LinearRegression()
            model.fit(X_tr, y_tr)
            y_pred_val = model.predict(X_val)
            r2_fold = r2_score(y_val, y_pred_val)
            scores.append(r2_fold)
            print(f"Fold {fold+1} R² (Log Scale): {r2_fold:.4f}")
            print(f"Fold {fold+1} y_mean: {y_val.mean():.4f}, y_std: {y_val.std():.4f}")
        print(f"Cross-validated R² (Log Scale): {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Step 8: Model Evaluation and Visualization
# Evaluate both models
evaluate_model(y_test, y_pred_simple, y_test_orig, y_pred_simple_orig, "Simple Linear Regression", X_test_simple)
evaluate_model(y_test, y_pred_multiple, y_test_orig, y_pred_multiple_orig, "Multiple Linear Regression", X_test)

# Visualize the results of Simple Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(X_test_simple, y_test_orig, color='blue', label='Actual')
plt.scatter(X_test_simple, y_pred_simple_orig, color='red', alpha=0.5, label='Predicted')
plt.xlabel('Log Area (scaled)')
plt.ylabel('Price (INR)')
plt.title('Simple Linear Regression: Log Area vs Price')
plt.legend()
plt.savefig('simple_regression_plot.png')
plt.show()

# Visualize Multiple Linear Regression: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, y_pred_multiple_orig, color='green', label='Predicted vs Actual')
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], color='red', linestyle='--', label='Ideal')
plt.xlabel('Actual Price (INR)')
plt.ylabel('Predicted Price (INR)')
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.legend()
plt.savefig('multiple_regression_plot.png')
plt.show()

# Residuals Plot for Multiple Linear Regression
residuals = y_test_orig - y_pred_multiple_orig
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_multiple_orig, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Price (INR)')
plt.ylabel('Residuals')
plt.title('Residual Plot: Multiple Linear Regression')
outlier_threshold = 3 * residuals.std()
outliers = np.abs(residuals) > outlier_threshold
plt.scatter(y_pred_multiple_orig[outliers], residuals[outliers], color='orange', label='Outliers')
plt.legend()
plt.savefig('residual_plot.png')
plt.show()

# Step 9: Result Interpretation and Data Saving
# Display the coefficients of both models for interpretation
print("\nSimple Linear Regression Coefficients (log scale):")
print(f"Intercept: {simple_lr.intercept_:.4f}")
print(f"Coefficient for log_area: {simple_lr.coef_[0]:.4f}")

print("\nMultiple Linear Regression Coefficients (log scale):")
for feature, coef in zip(selected_features, multiple_lr.coef_):
    print(f"{feature}: {coef:.4f}")

# Save the cleaned and processed dataset to a new CSV file
data.to_csv('processed_housing_data.csv', index=False)