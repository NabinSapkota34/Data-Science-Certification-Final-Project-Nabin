# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("--- Starting Data Science Pipeline ---")

# ==============================================================================
# Step 1: Data Collection and Exploration
# ==============================================================================
print("\n[Step 1] Loading and Exploring the Dataset...")
# Load the dataset from the CSV file
df = pd.read_csv('odb.csv')

# Initial exploration: Examine the structure and summary statistics
print("\nDataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nSummary Statistics:")
print(df.describe())
# We can see an 'Unnamed: 0' column which is just an index, we will drop it.
print("[Step 1] Completed.")


# ==============================================================================
# Step 2: Data Cleaning and Transformation
# ==============================================================================
print("\n[Step 2] Cleaning and Transforming Data...")
# Clean the 'HS' (Highest Score) column by removing '*' and converting to a numeric type
df['HS'] = df['HS'].astype(str).str.replace('*', '', regex=False)
df['HS'] = pd.to_numeric(df['HS'])

# *** FIX APPLIED HERE ***
# Clean the '4s' and '6s' columns by removing any non-digit characters and converting to numeric
df['4s'] = df['4s'].astype(str).str.replace(r'\D', '', regex=True)
df['6s'] = df['6s'].astype(str).str.replace(r'\D', '', regex=True)
df['4s'] = pd.to_numeric(df['4s'], errors='coerce').fillna(0).astype(int)
df['6s'] = pd.to_numeric(df['6s'], errors='coerce').fillna(0).astype(int)
# *** END OF FIX ***

# Feature Engineering: Create a 'Career_Length' feature from the 'Span' column
df['Start_Year'] = df['Span'].apply(lambda x: int(x.split('-')[0]))
df['End_Year'] = df['Span'].apply(lambda x: int(x.split('-')[1]))
df['Career_Length'] = df['End_Year'] - df['Start_Year']

# Prepare the final DataFrame for the model by dropping non-numeric or irrelevant columns
# Also dropping the 'Unnamed: 0' column which is a duplicate index.
df_model = df.drop(['Player', 'Span', 'Start_Year', 'End_Year', 'Unnamed: 0'], axis=1)

print("\nTransformed Data Head:")
print(df_model.head())
print("\nCleaned Data Types:")
df_model.info()
print("[Step 2] Completed.")


# ==============================================================================
# Step 3: Exploratory Data Analysis (EDA)
# ==============================================================================
print("\n[Step 3] Performing Exploratory Data Analysis...")
# Visualization 1: Distribution of the target variable 'Runs'
plt.figure(figsize=(10, 6))
sns.histplot(df_model['Runs'], kde=True, bins=30)
plt.title('Distribution of Player Career Runs')
plt.xlabel('Total Career Runs')
plt.ylabel('Frequency')
plt.savefig('runs_distribution.png') # Save the plot for documentation
print("   - Saved runs distribution plot to 'runs_distribution.png'")

# Visualization 2: Correlation Heatmap to understand feature relationships
plt.figure(figsize=(16, 12))
correlation_matrix = df_model.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features')
plt.savefig('correlation_heatmap.png') # Save the plot for documentation
print("   - Saved correlation heatmap to 'correlation_heatmap.png'")
print("Key Insight: 'Runs' has strong positive correlations with 'BF' (Balls Faced), 'Inns' (Innings), '100s', and '50s'.")
print("[Step 3] Completed.")

# ==============================================================================
# Step 4 & 5: Feature Selection and Model Development
# ==============================================================================
print("\n[Step 4 & 5] Selecting Features and Developing the Model...")
# Define features (X) and target (y)
X = df_model.drop('Runs', axis=1)
y = df_model['Runs']

# Data Splitting: Divide the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   - Training set size: {X_train.shape[0]} samples")
print(f"   - Testing set size: {X_test.shape[0]} samples")

# Algorithm Selection: We choose RandomForestRegressor, a powerful and versatile algorithm.
model = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True)

# Model Training
print("\n   - Training the RandomForestRegressor model...")
model.fit(X_train, y_train)
print("   - Model training complete.")
print("[Step 4 & 5] Completed.")

# ==============================================================================
# Step 6: Model Evaluation and Hyperparameter Tuning
# ==============================================================================
print("\n[Step 6] Evaluating the Model...")
# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate model performance using standard regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"   - Mean Absolute Error (MAE): {mae:.2f}")
print(f"   - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"   - R-squared (R²): {r2:.2f}")
print(f"   - Out-of-Bag (OOB) Score: {model.oob_score_:.2f} (A form of cross-validation)")

# Feature Importance Analysis
feature_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n   - Feature Importances:")
print(feature_importances)
print("Note: For a full project, hyperparameter tuning using GridSearchCV or RandomizedSearchCV would be the next step.")
print("[Step 6] Completed.")


# ==============================================================================
# Step 7: Model Deployment (Saving the Model)
# ==============================================================================
print("\n[Step 7] Saving the Trained Model...")
# Save the trained model to a file so our web app can use it
joblib.dump(model, 'odi_runs_predictor.pkl')
print("   - Model successfully saved as 'odi_runs_predictor.pkl'")
print("[Step 7] Completed.")

print("\n--- Data Science Pipeline Finished Successfully! ---")