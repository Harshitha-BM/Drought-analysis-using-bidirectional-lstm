import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
print("--- Phase 2: Data Preprocessing Script Starting ---")

# ==========================================================
# STEP 2.1: LOAD AND CLEAN THE DATA
# ==========================================================

# --- 1. Load your CSV file ---
# This path assumes your CSV is in the SAME folder as this script.
file_path = "Bagalkot_Drought_Indices_2015_2025_Optimized.csv"

# Check if file exists
if not os.path.exists(file_path):
    print(f"ERROR: File not found at {file_path}")
    print("Please make sure your CSV file is in the same folder as this script.")
else:
    print(f"Successfully found file: {file_path}")
    df = pd.read_csv(file_path)

    # --- 2. Clean the DataFrame ---
    # Drop the columns we don't need
    df_cleaned = df.drop(columns=['system:index', '.geo'])

    # Convert the 'date' column into a proper date object
    df_cleaned['date'] = pd.to_datetime(df_cleaned['date'])

    # Set the 'date' as the official index
    df_cleaned = df_cleaned.set_index('date')

    print(f"Original data loaded with {len(df_cleaned)} rows.")

    # ==========================================================
    # STEP 2.2: FILL GAPS (CRUCIAL STEP)
    # ==========================================================

    # --- 1. Create a full 5-day calendar from 2015 to 2025 ---
    full_date_range = pd.date_range(start='2015-01-01', end='2025-09-30', freq='5D')

    # --- 2. Re-index your data onto this full calendar ---
    df_resampled = df_cleaned.reindex(full_date_range)

    # --- 3. Fill the 'NaN' gaps using linear interpolation ---
    df_filled = df_resampled.interpolate(method='linear')

    # Just in case any gaps were at the very start
    df_filled = df_filled.dropna()

    print(f"Data resampled and interpolated. Total rows now: {len(df_filled)}")

    # ==========================================================
    # STEP 2.3: NORMALIZE THE DATA
    # ==========================================================

    # --- 1. Define your features ---
    features = ['LST', 'NDVI', 'TCI', 'VCI', 'VHI']
    data_to_scale = df_filled[features]

    # --- 2. Create the Scaler ---
    scaler = MinMaxScaler(feature_range=(0, 1))

    # --- 3. Fit and transform the data ---
    scaled_data = scaler.fit_transform(data_to_scale)

    print("Data successfully scaled between 0 and 1.")
    
    # You will need to save this scaler to use it in your final step
    # We will do this later, for now, we just use it.
    
    # ==========================================================
    # STEP 2.4: CREATE TIME-SERIES "WINDOWS"
    # ==========================================================

    def create_sequences(data, n_lookback, n_forecast, target_col_index):
        """Creates sequences of data for time-series forecasting."""
        X, y = [], []
        for i in range(len(data) - n_lookback - n_forecast + 1):
            lookback_slice = data[i : i + n_lookback]
            forecast_slice = data[i + n_lookback : i + n_lookback + n_forecast]
            
            X.append(lookback_slice)
            y.append(forecast_slice[:, target_col_index]) # Get just the VHI
            
        return np.array(X), np.array(y)

    # --- Define model parameters ---
    N_LOOKBACK = 12  # Use 12 past steps (12 * 5 days = 60 days)
    N_FORECAST = 1   # Predict 1 step in the future (5 days)
    TARGET_COL_INDEX = features.index('VHI') # This is 4

    # --- Create the sequences ---
    X, y = create_sequences(scaled_data, N_LOOKBACK, N_FORECAST, TARGET_COL_INDEX)

    print(f"\n--- Data Shapes Created ---")
    print(f"X (features) shape: {X.shape}") 
    print(f"y (labels) shape: {y.shape}")   

    # ==========================================================
    # STEP 2.5: SPLIT INTO TRAIN, VALIDATION, & TEST SETS
    # ==========================================================

    n_samples = X.shape[0]
    train_size = int(n_samples * 0.8)
    val_size = int(n_samples * 0.1)

    # --- Training Set (First 80%) ---
    X_train, y_train = X[:train_size], y[:train_size]

    # --- Validation Set (Next 10%) ---
    X_val, y_val = X[train_size : train_size + val_size], y[train_size : train_size + val_size]

    # --- Test Set (Final 10%) ---
    X_test, y_test = X[train_size + val_size :], y[train_size + val_size :]

    print(f"\n--- Final Split Shapes ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    print("\n--- Phase 2: Preprocessing Complete ---")
    print("You can now use X_train, y_train, X_val, etc., for training.")
# ==========================================================
# STEP 2.6: SAVE ALL PREPROCESSED DATA
# ==========================================================
print("\n--- Saving preprocessed data to files ---")

try:
    # Save the data arrays
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('X_val.npy', X_val)
    np.save('y_val.npy', y_val)
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)

    # Save the scaler object using joblib
    joblib.dump(scaler, 'scaler.gz')

    print("All data and scaler saved successfully.")
    print("--- Phase 2: Preprocessing Complete ---")

except Exception as e:
    print(f"An error occurred while saving data: {e}")

# (You can delete the old "--- Phase 2: Preprocessing Complete ---" line)

