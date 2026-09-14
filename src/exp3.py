import pandas as pd
import numpy as np
import warnings

# Preprocessing and Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.cross_decomposition import PLSRegression

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, RANSACRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Load Data ---
try:
    train_df = pd.read_csv('train_weather_data.csv')
    test_df = pd.read_csv('test_waether_data.csv')
    print("Training and testing data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please run the file check command (!ls -l) and re-upload your CSV files if they are missing.")
    exit()

# --- 2. Prepare Data (Define Target and Features) ---
# Define all target columns for team1 and team2 players
fantasy_points_attr = [
    "runs",
    "wickets",
    "4s",
    "6s",
    "catches",
    "maidens",
    "overs",
    "runs gave",
    "out",
    "ducks",
    "balls played",
    "lbw/bowled",
    "runout(indirect)",
    "runout(direct)",
    "fantasy points",
]
DROP_COLUMNS = [
    'team1_player1_fantasy points',
    'team1_player2_fantasy points',
    'team1_player3_fantasy points',
    'team1_player4_fantasy points',
    'team1_player5_fantasy points',
    'team2_player1_fantasy points',
    'team2_player2_fantasy points',
    'team2_player3_fantasy points',
    'team2_player4_fantasy points',
    'team2_player5_fantasy points'
]

TARGET_COLUMNS = []
for j in range(1, 3):
    for i in range(1, 12):
        for attr in fantasy_points_attr:
            TARGET_COLUMNS.append("team" + str(j) + "_" + "player" + str(i) + "_" + attr)
# Check if all target columns exist in both DataFrames
missing_in_train = [col for col in TARGET_COLUMNS if col not in train_df.columns]
missing_in_test = [col for col in TARGET_COLUMNS if col not in test_df.columns]
DROP_COLUMNS.extend(TARGET_COLUMNS)
if missing_in_train or missing_in_test:
    print("Error: Some target columns are missing:")
    if missing_in_train:
        print(f"  Missing in train_df: {missing_in_train}")
    if missing_in_test:
        print(f"  Missing in test_df: {missing_in_test}")
    print("\nPlease check the column names.")
    exit()
else:
    print(f"All {len(TARGET_COLUMNS)} target columns found successfully!")

# Separate features and targets for training data
existing_target_cols_train = [col for col in TARGET_COLUMNS if col in train_df.columns]
existing_drop_cols_train=[col for col in DROP_COLUMNS if col in train_df.columns]
X_train = train_df.drop(existing_drop_cols_train, axis=1)
print(X_train.columns.to_list())
y_train = train_df[existing_target_cols_train]

# Separate features and targets for test data
existing_target_cols_test = [col for col in TARGET_COLUMNS if col in test_df.columns]
existing_drop_cols_test=[col for col in DROP_COLUMNS if col in test_df.columns]
X_test = test_df.drop(existing_drop_cols_test, axis=1)
print(X_test.columns.to_list())
y_test = test_df[existing_target_cols_test]

# Ensure X_test has the same columns as X_train in the same order
train_cols = X_train.columns
X_test = X_test.reindex(columns=train_cols, fill_value=0)

# Verify shapes
print(f"\nX_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Number of target columns: {len(existing_target_cols_train)}")


# --- 3. Define the Evaluation Function ---
def train_and_evaluate_on_test_set(X_train, y_train, X_test, y_test):
    """
    Trains multiple regression models and evaluates them on test data.
    Handles multiple output targets (all players' fantasy points).
    """
    print(f"\nTraining on {X_train.shape[0]} samples and testing on {X_test.shape[0]} samples.")

    # Identify feature types, excluding identifiers
    categorical_features = [col for col in X_train.select_dtypes(include=['object', 'category']).columns 
                           if 'player' not in col and col != 'date']
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()

    # Define the base preprocessor
    base_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ], remainder='drop')

    # Define the dictionary of models
    # Note: Models that natively support multi-output don't need MultiOutputRegressor
    # Models that don't support multi-output natively need MultiOutputRegressor wrapper
    models = {
        # Linear models (native multi-output support)
        "Lasso": Lasso(random_state=42),
        "Ridge": Ridge(random_state=42),
        "ElasticNet": ElasticNet(random_state=42),
        
        # Robust linear models (need MultiOutputRegressor)
        "Huber": MultiOutputRegressor(HuberRegressor()),
        "RANSAC": MultiOutputRegressor(RANSACRegressor(random_state=42)),
        
        # Ensemble models (native multi-output support)
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100),
        
        # Models that need MultiOutputRegressor
        "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor(random_state=42, n_estimators=100)),
        "Support Vector Regressor": MultiOutputRegressor(SVR()),
        "CatBoost": MultiOutputRegressor(CatBoostRegressor(verbose=0, random_state=42, iterations=100)),
        "XGBoost": MultiOutputRegressor(XGBRegressor(random_state=42, n_estimators=100)),
        
        # Gaussian Process (need MultiOutputRegressor)
        "Gaussian Process Regressor": MultiOutputRegressor(
            GaussianProcessRegressor(
                kernel=C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)), 
                random_state=42
            )
        ),
        
        # Dimensionality reduction models (need special handling)
        "Partial Least Squares": PLSRegression(n_components=min(10, len(numerical_features))),
        
        "Principal Component Regression": Pipeline(steps=[
            ('preprocessor', base_preprocessor),
            ('pca', PCA(n_components=min(20, len(numerical_features)))),
            ('regressor', LinearRegression())
        ]),
        
        "ICA Regression": Pipeline(steps=[
            ('preprocessor', base_preprocessor),
            ('ica', FastICA(n_components=min(20, len(numerical_features)), random_state=42, max_iter=1000)),
            ('regressor', LinearRegression())
        ])
    }

    results = {}
    # Models that are already a full pipeline
    full_pipeline_models = ["Principal Component Regression", "ICA Regression"]

    for name, model in models.items():
        print(f"--- Training {name} ---")
        
        try:
            if name in full_pipeline_models:
                pipeline = model
            else:
                # Wrap standard models in the base preprocessor
                pipeline = Pipeline(steps=[('preprocessor', base_preprocessor), ('regressor', model)])

            # Train the model on the entire training set
            pipeline.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = pipeline.predict(X_test)

            # Calculate Mean Relative Error (avoiding division by zero)
            y_test_array = np.array(y_test)
            y_pred_array = np.array(y_pred)
            
            # Mask to avoid division by zero
            non_zero_mask = y_test_array != 0
            relative_errors = np.abs((y_test_array[non_zero_mask] - y_pred_array[non_zero_mask]) / y_test_array[non_zero_mask])
            mre = np.mean(relative_errors) if len(relative_errors) > 0 else np.nan
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = mre * 100 if not np.isnan(mre) else np.nan

            # Calculate and store performance metrics (averaged across all outputs)
            results[name] = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R2': r2_score(y_test, y_pred),
                'MRE': mre,
                'MAPE (%)': mape
            }
            
            print(f"  MAE: {results[name]['MAE']:.4f}, MAPE: {results[name]['MAPE (%)']:.2f}%")
            
        except Exception as e:
            print(f"  Error training {name}: {str(e)}")
            results[name] = {
                'MAE': np.nan,
                'MSE': np.nan,
                'RMSE': np.nan,
                'R2': np.nan,
                'MRE': np.nan,
                'MAPE (%)': np.nan
            }
    
    return results

# --- 4. Run the Full Process and Display Results ---
model_results = train_and_evaluate_on_test_set(X_train, y_train, X_test, y_test)

# Convert the results dictionary to a DataFrame for clear comparison
results_df = pd.DataFrame(model_results).T

# Remove rows with NaN values (failed models)
results_df_clean = results_df.dropna()

print("\n" + "="*80)
print("--- Model Performance on Test Data (Averaged Across All Players) ---")
print("="*80)

if len(results_df_clean) > 0:
    # Sort results by MAPE (lower is better) to find the best model
    print("\nSorted by MAPE (%):")
    print(results_df_clean.sort_values(by='MAPE (%)', ascending=True).to_string())

    print("\n" + "="*80)
    print("\nSorted by MAE:")
    print(results_df_clean.sort_values(by='MAE', ascending=True).to_string())

    # Display best model
    best_model_mape = results_df_clean['MAPE (%)'].idxmin()
    best_model_mae = results_df_clean['MAE'].idxmin()
    print("\n" + "="*80)
    print(f"Best Model by MAPE: {best_model_mape} ({results_df_clean.loc[best_model_mape, 'MAPE (%)']:.2f}%)")
    print(f"Best Model by MAE: {best_model_mae} ({results_df_clean.loc[best_model_mae, 'MAE']:.4f})")
    print("="*80)
else:
    print("\nNo models completed successfully.")