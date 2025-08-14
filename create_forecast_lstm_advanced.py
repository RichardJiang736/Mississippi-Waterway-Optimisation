import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import warnings
warnings.filterwarnings('ignore')

# Constants - ADVANCED PARAMETERS
RANDOM_SEED = 42
SEQUENCE_LENGTH = 10
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Alternative loss functions that encourage variation
def quantile_loss(quantile=0.5):
    """Quantile loss - can target different quantiles"""
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return K.mean(K.maximum(quantile * error, (quantile - 1) * error))
    return loss

def huber_loss_custom(delta=1.0):
    """Huber loss - less sensitive to outliers than MSE"""
    def loss(y_true, y_pred):
        error = y_true - y_pred
        condition = K.abs(error) <= delta
        squared_loss = 0.5 * K.square(error)
        linear_loss = delta * K.abs(error) - 0.5 * K.square(delta)
        return tf.where(condition, squared_loss, linear_loss)
    return loss

def asymmetric_loss(over_penalty=1.0, under_penalty=1.5):
    """Asymmetric loss - different penalties for over/under prediction"""
    def loss(y_true, y_pred):
        error = y_true - y_pred
        over_pred = K.maximum(0.0, -error)  # Positive when over-predicting
        under_pred = K.maximum(0.0, error)  # Positive when under-predicting
        return K.mean(over_penalty * over_pred + under_penalty * under_pred)
    return loss

def variation_encouraging_loss(base_loss_weight=0.7, var_penalty_weight=0.3):
    """Loss that explicitly encourages variation"""
    def loss(y_true, y_pred):
        # Base MAE loss
        mae = K.mean(K.abs(y_true - y_pred))
        
        # Encourage variation by penalizing low variance in predictions
        pred_std = K.std(y_pred)
        true_std = K.std(y_true)
        
        # Penalty when prediction std is much lower than true std
        std_ratio = K.maximum(0.0, true_std - pred_std) / (true_std + 1e-8)
        variance_penalty = K.square(std_ratio)
        
        return base_loss_weight * mae + var_penalty_weight * variance_penalty
    return loss

print("Loading and preprocessing data...")

# Load data
df = pd.read_csv('LA_import_export.csv').drop('Unnamed: 0', axis=1)
df = df.sort_values(['Import or Export', 'Date'])
df['Date'] = pd.to_datetime(df['Date'])
df[['predicted_hscode', 'sea_distance_km']] = df[['predicted_hscode', 'sea_distance_km']].fillna(0)

# Container type mapping function
def map_container_type(code):
    if pd.isna(code):
        return np.nan
    code = str(code)
    if code.startswith('45'):
        if 'R' in code:
            return "40-foot high cube 9'6\" reefer container"
        else:
            return "40-foot high cube 9'6\" dry general usage container"
    elif code.startswith('42'):
        if 'R' in code:
            return "40-foot 8'6\" height reefer container"
        else:
            return "40-foot 8'6\" height dry general usage container"
    elif code.startswith('40'):
        if 'R' in code:
            return "40-foot standard reefer container"
        else:
            return "40-foot standard dry general usage container"
    elif code.startswith('20'):
        if 'R' in code:
            return "20-foot reefer container"
        else:
            return "20-foot dry general usage container"
    elif 'B' in code:
        return "Bulk container"
    else:
        return "Other container type"

df['ContainerTypeMerged'] = df['containerType'].apply(map_container_type)

print("Creating weekly aggregated data...")

# Convert to weekly data
df['Date'] = pd.to_datetime(df['Date']).dt.to_period('W').dt.to_timestamp()

# Filter for specific container type and import (0)
container_type = "40-foot high cube 9'6\" dry general usage container"
df_filtered = df[
    (df['ContainerTypeMerged'] == container_type) &
    (df['Import or Export'] == 0)
].copy()

print(f"Working with container type: {container_type}")
print(f"Filtered data shape: {df_filtered.shape}")

# Aggregate to weekly level
agg_dict = {
    'noOfContainers': 'sum',
    'shippingWeightKg': 'mean',
    'sea_distance_km': 'mean',
    'TMAX': 'mean', 'TMIN': 'mean', 'TMPC': 'mean',
    'PCPN': 'mean', 'PDSI': 'mean', 'PHDI': 'mean', 'ZNDX': 'mean'
}

df_agg = df_filtered.groupby('Date').agg(agg_dict).reset_index()

# Add most common HS code for each week
df_filtered['predicted_hscode'] = df_filtered['predicted_hscode'].fillna(0).astype(int).apply(lambda x: f"{x:02d}")
hs_mode = df_filtered.groupby('Date')['predicted_hscode'].apply(
    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else '27'
).reset_index()
df_agg = df_agg.merge(hs_mode, on='Date', how='left')
df_agg['hs_2digit'] = df_agg['predicted_hscode'].astype(str).str[:2]

print(f"Aggregated data shape: {df_agg.shape}")
print(f"Weekly container stats:")
print(df_agg['noOfContainers'].describe())

# Sort by date and ensure proper index
df_agg = df_agg.sort_values('Date').reset_index(drop=True)

# Create enhanced features
print("Creating enhanced features...")
for lag in range(1, 9):
    df_agg[f'lag{lag}'] = df_agg['noOfContainers'].shift(lag)

# Add volatility and trend features
for window in [3, 5, 8]:
    df_agg[f'rolling_mean_{window}'] = df_agg['noOfContainers'].rolling(window=window, min_periods=1).mean()
    df_agg[f'rolling_std_{window}'] = df_agg['noOfContainers'].rolling(window=window, min_periods=1).std().fillna(0)
    df_agg[f'rolling_min_{window}'] = df_agg['noOfContainers'].rolling(window=window, min_periods=1).min()
    df_agg[f'rolling_max_{window}'] = df_agg['noOfContainers'].rolling(window=window, min_periods=1).max()

# Add percentage changes and momentum
df_agg['pct_change_1'] = df_agg['noOfContainers'].pct_change(1).fillna(0)
df_agg['pct_change_4'] = df_agg['noOfContainers'].pct_change(4).fillna(0)
df_agg['momentum_3'] = df_agg['noOfContainers'].diff(3).fillna(0)

# Add seasonal features
df_agg['Month'] = df_agg['Date'].dt.month
df_agg['Quarter'] = df_agg['Date'].dt.quarter
df_agg['Week'] = df_agg['Date'].dt.isocalendar().week
df_agg['sin_month'] = np.sin(2 * np.pi * df_agg['Month'] / 12)
df_agg['cos_month'] = np.cos(2 * np.pi * df_agg['Month'] / 12)
df_agg['sin_week'] = np.sin(2 * np.pi * df_agg['Week'] / 52)
df_agg['cos_week'] = np.cos(2 * np.pi * df_agg['Week'] / 52)

# Drop rows with NaN lag features
df_agg = df_agg.dropna().reset_index(drop=True)

print(f"After removing NaN rows: {len(df_agg)} samples")

print("Preparing enhanced features for LSTM...")

# Enhanced feature columns
num_cols = ['shippingWeightKg', 'sea_distance_km', 'Month', 'Week', 'Quarter',
           'lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8',
           'rolling_mean_3', 'rolling_std_3', 'rolling_min_3', 'rolling_max_3',
           'rolling_mean_5', 'rolling_std_5', 'rolling_min_5', 'rolling_max_5',
           'rolling_mean_8', 'rolling_std_8', 'rolling_min_8', 'rolling_max_8',
           'pct_change_1', 'pct_change_4', 'momentum_3',
           'sin_month', 'cos_month', 'sin_week', 'cos_week']
cat_cols = ['hs_2digit']
weather_cols = ['TMAX', 'TMIN', 'TMPC', 'PCPN']

# Combine all feature columns
feature_cols = num_cols + cat_cols + weather_cols

# One-hot encode categorical features
df_features = df_agg[feature_cols].copy()
df_cat = df_features[cat_cols]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_encoded = encoder.fit_transform(df_cat)
encoded_col_names = encoder.get_feature_names_out(df_cat.columns)
cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoded_col_names, index=df_cat.index)

# Combine numerical and categorical features
X = pd.concat([df_features.drop(cat_cols, axis=1), cat_encoded_df], axis=1)

# Target variable - NO LOG TRANSFORMATION
y = df_agg['noOfContainers'].copy()

# Fill any NaN values
X = X.fillna(X.mean())
y = y.fillna(y.mean())

print(f"Feature matrix shape: {X.shape}")
print(f"Target stats: mean={y.mean():.1f}, std={y.std():.1f}")

# Use RobustScaler to preserve outliers and variation better
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

print("Creating sequences for LSTM...")

def create_sequences(X, y, sequence_length):
    """Create sequences for LSTM training"""
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)

# Create sequences
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQUENCE_LENGTH)

print(f"Sequence shape: X_seq={X_seq.shape}, y_seq={y_seq.shape}")

# Split data temporally (80% train, 20% test)
split_point = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_point], X_seq[split_point:]
y_train, y_test = y_seq[:split_point], y_seq[split_point:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# Create models with different loss functions
def create_model(input_shape, model_id, loss_func, loss_name):
    """Create a model with specified loss function"""
    np.random.seed(RANDOM_SEED + model_id)
    tf.random.set_seed(RANDOM_SEED + model_id)
    
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.02),  # Even lower dropout
        
        Bidirectional(LSTM(32, return_sequences=True)),
        Dropout(0.02),
        
        LSTM(24, return_sequences=False),
        Dropout(0.02),
        
        Dense(64, activation='relu'),
        Dropout(0.02),
        Dense(32, activation='relu'),
        Dropout(0.02),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Compile with specified loss function
    model.compile(
        optimizer=Adam(learning_rate=0.005),  # Higher learning rate
        loss=loss_func, 
        metrics=['mae']
    )
    
    return model

print("Building ADVANCED multi-loss ensemble LSTM models...")

# Define different loss functions to try
loss_configs = [
    ('MAE', 'mae'),
    ('Huber', huber_loss_custom(delta=0.5)),
    ('Quantile_0.7', quantile_loss(0.7)),  # Target 70th percentile
    ('Asymmetric', asymmetric_loss(over_penalty=0.8, under_penalty=1.5)),
    ('Variation_Encouraging', variation_encouraging_loss())
]

models = []
histories = []
predictions_train = []
predictions_test = []
loss_names = []

for i, (loss_name, loss_func) in enumerate(loss_configs):
    print(f"\nTraining model {i+1}/{len(loss_configs)} with {loss_name} loss...")
    
    model = create_model((SEQUENCE_LENGTH, X_seq.shape[2]), i, loss_func, loss_name)
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=300,  # More epochs
        batch_size=2,  # Even smaller batch size
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Generate predictions
    y_pred_train_scaled = model.predict(X_train, verbose=0)
    y_pred_test_scaled = model.predict(X_test, verbose=0)
    
    # Store results
    models.append(model)
    histories.append(history)
    predictions_train.append(scaler_y.inverse_transform(y_pred_train_scaled).flatten())
    predictions_test.append(scaler_y.inverse_transform(y_pred_test_scaled).flatten())
    loss_names.append(loss_name)

# Calculate ensemble predictions
y_pred_train = np.mean(predictions_train, axis=0)
y_pred_test = np.mean(predictions_test, axis=0)

# Get actual values with proper indexing
y_train_actual = y.iloc[SEQUENCE_LENGTH:SEQUENCE_LENGTH+len(X_train)].values
y_test_actual = y.iloc[SEQUENCE_LENGTH+len(X_train):SEQUENCE_LENGTH+len(X_train)+len(X_test)].values

print(f"\nMulti-Loss Ensemble Results:")
print(f"Train predictions: mean={y_pred_train.mean():.1f}, std={y_pred_train.std():.1f}")
print(f"Test predictions: mean={y_pred_test.mean():.1f}, std={y_pred_test.std():.1f}")
print(f"Train actual: mean={y_train_actual.mean():.1f}, std={y_train_actual.std():.1f}")
print(f"Test actual: mean={y_test_actual.mean():.1f}, std={y_test_actual.std():.1f}")

# Calculate ensemble performance metrics for plotting
print("Creating multi-loss ensemble performance plots...")

# Calculate ensemble loss over epochs (reconstruct from predictions)
def calculate_ensemble_loss(predictions_list, actual_values):
    """Calculate MSE loss for ensemble predictions"""
    ensemble_pred = np.mean(predictions_list, axis=0)
    return np.mean((ensemble_pred - actual_values) ** 2)

def calculate_ensemble_mae(predictions_list, actual_values):
    """Calculate MAE for ensemble predictions"""
    ensemble_pred = np.mean(predictions_list, axis=0)
    return np.mean(np.abs(ensemble_pred - actual_values))

# Get the maximum number of epochs across all models
max_epochs = max(len(hist.history['loss']) for hist in histories)

# Create ensemble performance plot
plt.figure(figsize=(16, 6))

# Plot 1: Multi-Loss Ensemble Loss
plt.subplot(1, 3, 1)

# Plot individual model losses (lighter lines)
colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow', 'lightpink']
for i, (history, loss_name, color) in enumerate(zip(histories, loss_names, colors)):
    plt.plot(history.history['loss'], color=color, alpha=0.6, linewidth=1, 
             label=f'{loss_name} Train')
    plt.plot(history.history['val_loss'], color=color, alpha=0.6, linewidth=1, 
             linestyle='--', label=f'{loss_name} Val')

# Calculate weighted ensemble loss trajectory
ensemble_train_loss = []
ensemble_val_loss = []

# For ensemble loss, we need to approximate since we can't easily reconstruct epoch-by-epoch
# Instead, show the best performing individual models prominently
best_train_model = min(range(len(histories)), key=lambda i: histories[i].history['loss'][-1])
best_val_model = min(range(len(histories)), key=lambda i: histories[i].history['val_loss'][-1])

plt.plot(histories[best_train_model].history['loss'], color='darkgreen', linewidth=3, 
         label=f'Best Train ({loss_names[best_train_model]})')
plt.plot(histories[best_val_model].history['val_loss'], color='darkblue', linewidth=3, 
         label=f'Best Val ({loss_names[best_val_model]})')

plt.title('Multi-Loss Ensemble Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: MAE Comparison
plt.subplot(1, 3, 2)

# Show ensemble MAE performance
ensemble_train_mae = calculate_ensemble_mae(predictions_train, y_train_actual)
ensemble_test_mae = calculate_ensemble_mae(predictions_test, y_test_actual)

# Plot individual model final MAEs
individual_train_maes = [mean_absolute_error(y_train_actual, pred) for pred in predictions_train]
individual_test_maes = [mean_absolute_error(y_test_actual, pred) for pred in predictions_test]

x_pos = np.arange(len(loss_names))
plt.bar(x_pos - 0.2, individual_train_maes, 0.3, label='Individual Train MAE', alpha=0.7, color='lightblue')
plt.bar(x_pos + 0.2, individual_test_maes, 0.3, label='Individual Test MAE', alpha=0.7, color='lightcoral')

# Add ensemble performance
plt.axhline(y=ensemble_train_mae, color='darkgreen', linewidth=3, label=f'Ensemble Train MAE: {ensemble_train_mae:.1f}')
plt.axhline(y=ensemble_test_mae, color='darkblue', linewidth=3, label=f'Ensemble Test MAE: {ensemble_test_mae:.1f}')

plt.title('MAE Performance: Individual vs Ensemble')
plt.xlabel('Loss Function')
plt.ylabel('MAE')
plt.xticks(x_pos, loss_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Variance Preservation
plt.subplot(1, 3, 3)

# Calculate variance for each model
individual_train_vars = [np.var(pred) for pred in predictions_train]
individual_test_vars = [np.var(pred) for pred in predictions_test]
ensemble_train_var = np.var(y_pred_train)
ensemble_test_var = np.var(y_pred_test)
actual_train_var = np.var(y_train_actual)
actual_test_var = np.var(y_test_actual)

plt.bar(x_pos - 0.2, individual_train_vars, 0.3, label='Individual Train Variance', alpha=0.7, color='lightgreen')
plt.bar(x_pos + 0.2, individual_test_vars, 0.3, label='Individual Test Variance', alpha=0.7, color='lightcoral')

# Add ensemble and actual variance lines
plt.axhline(y=ensemble_train_var, color='darkgreen', linewidth=3, label=f'Ensemble Train Var: {ensemble_train_var:.0f}')
plt.axhline(y=ensemble_test_var, color='darkblue', linewidth=3, label=f'Ensemble Test Var: {ensemble_test_var:.0f}')
plt.axhline(y=actual_train_var, color='red', linewidth=2, linestyle=':', label=f'Actual Train Var: {actual_train_var:.0f}')
plt.axhline(y=actual_test_var, color='orange', linewidth=2, linestyle=':', label=f'Actual Test Var: {actual_test_var:.0f}')

plt.title('Variance Preservation: Individual vs Ensemble')
plt.xlabel('Loss Function')
plt.ylabel('Variance')
plt.xticks(x_pos, loss_names, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Creating future forecasts with multi-loss ensemble...")

# Multi-step forecasting
forecast_steps = 13
forecast_values = []
forecast_dates = pd.date_range(start=df_agg['Date'].max() + pd.Timedelta(weeks=1), 
                              periods=forecast_steps, freq='W')

# Use the last sequence from the dataset as starting point
last_sequence = X_scaled[-SEQUENCE_LENGTH:].copy()

# Most common HS code
most_common_hs = df_agg['hs_2digit'].mode().iloc[0]

print(f"Generating {forecast_steps}-step multi-loss ensemble forecast...")

for step in range(forecast_steps):
    print(f"  Step {step+1}/{forecast_steps}...")
    
    # Get predictions from all models
    step_predictions = []
    for model in models:
        next_pred_scaled = model.predict(last_sequence.reshape(1, SEQUENCE_LENGTH, -1), verbose=0)[0, 0]
        next_pred = scaler_y.inverse_transform([[next_pred_scaled]])[0, 0]
        step_predictions.append(next_pred)
    
    # Weighted ensemble prediction (give more weight to variation-encouraging models)
    weights = [1.0, 1.0, 1.2, 1.3, 1.5]  # Higher weight for quantile, asymmetric, variation models
    weights = np.array(weights) / np.sum(weights)
    next_pred = np.average(step_predictions, weights=weights)
    forecast_values.append(next_pred)
    
    # Create next feature vector for updating sequence
    future_date = forecast_dates[step]
    
    # Time features
    month = future_date.month
    week = future_date.isocalendar().week
    quarter = future_date.quarter
    
    # Use historical averages for weather and other features
    avg_features = X.iloc[-10:].mean()
    
    # Create next feature vector
    next_features = avg_features.copy()
    next_features['Month'] = month
    next_features['Week'] = week
    next_features['Quarter'] = quarter
    next_features['sin_month'] = np.sin(2 * np.pi * month / 12)
    next_features['cos_month'] = np.cos(2 * np.pi * month / 12)
    next_features['sin_week'] = np.sin(2 * np.pi * week / 52)
    next_features['cos_week'] = np.cos(2 * np.pi * week / 52)
    
    # Update lag features with prediction
    if step == 0:
        # Use actual last values for first prediction
        last_containers = y.tail(8).values
        for i, lag in enumerate(['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8']):
            next_features[lag] = last_containers[-(i+1)]
    else:
        # Use predicted values for subsequent predictions
        recent_preds = forecast_values[-min(8, len(forecast_values)):]
        for i, lag in enumerate(['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag8']):
            if i < len(recent_preds):
                next_features[lag] = recent_preds[-(i+1)]
    
    # Update rolling features
    if len(forecast_values) >= 3:
        recent_3 = forecast_values[-3:]
        next_features['rolling_mean_3'] = np.mean(recent_3)
        next_features['rolling_std_3'] = np.std(recent_3) if len(recent_3) > 1 else 0
        next_features['rolling_min_3'] = np.min(recent_3)
        next_features['rolling_max_3'] = np.max(recent_3)
    
    if len(forecast_values) >= 5:
        recent_5 = forecast_values[-5:]
        next_features['rolling_mean_5'] = np.mean(recent_5)
        next_features['rolling_std_5'] = np.std(recent_5) if len(recent_5) > 1 else 0
        next_features['rolling_min_5'] = np.min(recent_5)
        next_features['rolling_max_5'] = np.max(recent_5)
    
    if len(forecast_values) >= 1:
        next_features['pct_change_1'] = (next_pred - forecast_values[-1]) / forecast_values[-1] if len(forecast_values) > 0 else 0
    
    if len(forecast_values) >= 4:
        next_features['pct_change_4'] = (next_pred - forecast_values[-4]) / forecast_values[-4]
        next_features['momentum_3'] = next_pred - forecast_values[-3]
    
    # Scale the features
    next_features_scaled = scaler_X.transform([next_features.values])[0]
    
    # Update sequence by removing first element and adding new one
    last_sequence = np.vstack([last_sequence[1:], next_features_scaled])

print("Creating ADVANCED plot...")

# Create plot with correct date alignment
plt.figure(figsize=(16, 8))

# Calculate date ranges
train_start_idx = SEQUENCE_LENGTH
train_end_idx = SEQUENCE_LENGTH + len(X_train)
test_start_idx = train_end_idx
test_end_idx = SEQUENCE_LENGTH + len(X_seq)

train_dates = df_agg['Date'].iloc[train_start_idx:train_end_idx]
test_dates = df_agg['Date'].iloc[test_start_idx:test_end_idx]

print(f"Date alignment:")
print(f"  Train predictions: {len(train_dates)} dates, {len(y_pred_train)} predictions")
print(f"  Test predictions: {len(test_dates)} dates, {len(y_pred_test)} predictions")

# Plot actual data
plt.plot(df_agg['Date'], y, color='gray', alpha=0.8, label='Actual', 
         linestyle='-', linewidth=2)

# Plot individual model predictions for comparison
colors = ['lightgreen', 'lightblue', 'lightcoral', 'lightyellow', 'lightpink']
for i, (pred_train, pred_test, loss_name, color) in enumerate(zip(predictions_train, predictions_test, loss_names, colors)):
    if i < 3:  # Only show first 3 to avoid cluttering
        plt.plot(train_dates, pred_train, color=color, alpha=0.5, linewidth=1, 
                label=f'{loss_name} Train')
        plt.plot(test_dates, pred_test, color=color, alpha=0.5, linewidth=1, linestyle='--',
                label=f'{loss_name} Test')

# Plot ensemble predictions
plt.plot(train_dates, y_pred_train, color='green', alpha=0.9, label='Multi-Loss Ensemble Train', 
         linestyle='-', linewidth=3)
plt.plot(test_dates, y_pred_test, color='blue', alpha=0.9, label='Multi-Loss Ensemble Test', 
         linestyle='-', linewidth=3)

# Plot forecasts
plt.plot(forecast_dates, forecast_values, color='orange', alpha=0.9, 
         label='Multi-Loss Ensemble Forecast (13 steps)', linestyle='--', linewidth=3)

# Add vertical line to separate historical and forecast data
last_date = df_agg['Date'].iloc[-1]
plt.axvline(x=last_date, color='red', linestyle=':', alpha=0.7, linewidth=1, label='Forecast Start')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title(f"Advanced Multi-Loss LSTM Container Import Predictions vs Actual\n{container_type}")
plt.ylabel("Number of Containers (Weekly Total)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate performance metrics
train_mae = mean_absolute_error(y_train_actual, y_pred_train)
test_mae = mean_absolute_error(y_test_actual, y_pred_test)
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test))

print(f"\nPerformance Metrics:")
print(f"Ensemble Train MAE: {train_mae:.1f} containers")
print(f"Ensemble Test MAE: {test_mae:.1f} containers")
print(f"Ensemble Train RMSE: {train_rmse:.1f} containers")
print(f"Ensemble Test RMSE: {test_rmse:.1f} containers")

# Individual model performance
print(f"\nIndividual Model Performance:")
for i, (pred_train, pred_test, loss_name) in enumerate(zip(predictions_train, predictions_test, loss_names)):
    train_mae_ind = mean_absolute_error(y_train_actual, pred_train)
    test_mae_ind = mean_absolute_error(y_test_actual, pred_test)
    train_std_ind = np.std(pred_train)
    test_std_ind = np.std(pred_test)
    print(f"  {loss_name}: Train MAE={train_mae_ind:.1f}, Test MAE={test_mae_ind:.1f}, "
          f"Train Std={train_std_ind:.1f}, Test Std={test_std_ind:.1f}")

# Variance comparison
print(f"\nVariance Analysis:")
print(f"Actual train variance: {np.var(y_train_actual):.1f}")
print(f"Ensemble train variance: {np.var(y_pred_train):.1f}")
print(f"Actual test variance: {np.var(y_test_actual):.1f}")
print(f"Ensemble test variance: {np.var(y_pred_test):.1f}")

print(f"\nFuture forecast stats:")
print(pd.Series(forecast_values).describe())

print(f"\nAdvanced Multi-Loss LSTM Model Configuration:")
print(f"- Ensemble of {len(loss_configs)} models with different loss functions:")
for loss_name in loss_names:
    print(f"  * {loss_name}")
print(f"- Sequence length: {SEQUENCE_LENGTH} weeks")
print(f"- Weighted ensemble (higher weight for variation-encouraging losses)")
print(f"- RobustScaler to preserve outliers and variation")
print(f"- Enhanced features: {X.shape[1]} total features")
print(f"- Very low dropout (0.02) to preserve variation")
print(f"- Small batch size (2) for precise learning")
print(f"- High learning rate (0.005)")

print(f"\nForecast period: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")

print("\nForecast dates:")
for i, (date, pred) in enumerate(zip(forecast_dates, forecast_values)):
    print(f"  Step {i+1}: {date.strftime('%Y-%m-%d')} - {pred:.0f} containers") 