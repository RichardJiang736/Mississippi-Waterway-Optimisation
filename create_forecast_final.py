import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier

# Constants
RANDOM_SEED = 42

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

# Aggregate to weekly level - KEY STEP!
agg_dict = {
    'noOfContainers': 'sum',  # Sum containers per week
    'shippingWeightKg': 'mean',  # Average weight
    'sea_distance_km': 'mean',  # Average distance
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

# Create lag features using AGGREGATED weekly data
print("Creating lagged features...")
for lag in range(1, 5):
    df_agg[f'lag{lag}'] = df_agg['noOfContainers'].shift(lag)

# Drop rows with NaN lag features (first 4 rows)
df_agg = df_agg.dropna().reset_index(drop=True)

print(f"After removing NaN rows: {len(df_agg)} samples")
print("Last few weekly container values for lags:")
print(df_agg[['Date', 'noOfContainers', 'lag1', 'lag2', 'lag3', 'lag4']].tail())

print("Preparing features...")

# Create time series features with proper DatetimeIndex
date_index = pd.DatetimeIndex(df_agg['Date'])
# Infer the frequency 
date_index = date_index.to_period('W').to_timestamp()
fourier = CalendarFourier(freq='W', order=1)
dp = DeterministicProcess(
    index=date_index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True
)
X_time = dp.in_sample()

# Add time features
df_agg['Year'] = df_agg['Date'].dt.year
df_agg['Month'] = df_agg['Date'].dt.month
df_agg['Week'] = df_agg['Date'].dt.isocalendar().week

# Feature preparation
time_cols = ['Year', 'Month', 'Week']
num_cols = ['shippingWeightKg', 'sea_distance_km', 'Year', 'lag1', 'lag2', 'lag3', 'lag4']
cat_cols = ['hs_2digit']
weather_cols = ['TMAX', 'TMIN', 'TMPC', 'PCPN', 'PDSI', 'PHDI', 'ZNDX']

# Log transform numerical features
df_agg[num_cols] = np.log1p(df_agg[num_cols])

# Create feature matrices
X_1 = X_time.copy()
X_1['Year'] = np.log1p(df_agg['Year'])
X_1['Month'] = df_agg['Month']
X_1['Week'] = df_agg['Week']

X_2 = df_agg[num_cols + cat_cols + weather_cols].copy()

# One-hot encode categorical features
X_2_cat = X_2[cat_cols]
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_2_cat_encoded = encoder.fit_transform(X_2_cat)
encoded_col_names = encoder.get_feature_names_out(X_2_cat.columns)
X_2_cat_encoded_df = pd.DataFrame(X_2_cat_encoded, columns=encoded_col_names, index=X_2_cat.index)

X_2 = pd.concat([X_2.drop(cat_cols, axis=1), X_2_cat_encoded_df], axis=1)

# Target variable
y = df_agg['noOfContainers'].copy()
y_log = np.log1p(y)

# Check for NaN values before training
print("Checking for NaN values:")
print(f"X_1 has NaN: {X_1.isna().any().any()}")
print(f"X_2 has NaN: {X_2.isna().any().any()}")
print(f"y_log has NaN: {y_log.isna().any()}")

# Fill any remaining NaN values
X_1 = X_1.fillna(0)
X_2 = X_2.fillna(0)
y_log = y_log.fillna(y_log.mean())

print(f"After filling NaN:")
print(f"X_1 has NaN: {X_1.isna().any().any()}")
print(f"X_2 has NaN: {X_2.isna().any().any()}")
print(f"y_log has NaN: {y_log.isna().any()}")

print(f"Target (y) stats - original: mean={y.mean():.1f}, std={y.std():.1f}")
print(f"Target (y) stats - log: mean={y_log.mean():.3f}, std={y_log.std():.3f}")

# Split data temporally
split_point = int(len(X_1) * 0.8)
X_1_train, X_1_test = X_1.iloc[:split_point], X_1.iloc[split_point:]
X_2_train, X_2_test = X_2.iloc[:split_point], X_2.iloc[split_point:]
y_train, y_test = y_log.iloc[:split_point], y_log.iloc[split_point:]

print(f"Train size: {len(X_1_train)}, Test size: {len(X_1_test)}")

print("Training model...")

# Define and train hybrid model
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.fitted_ = False

    def fit(self, X_1, X_2, y):
        self.model_1.fit(X_1, y)
        residuals = y - self.model_1.predict(X_1)
        self.model_2.fit(X_2, residuals)
        self.fitted_ = True
        return self

    def predict(self, X_1, X_2):
        if not self.fitted_:
            raise Exception("Model not fitted yet.")
        pred_1 = self.model_1.predict(X_1)
        pred_2 = self.model_2.predict(X_2)
        return pred_1 + pred_2

hybrid = BoostedHybrid(
    model_1=LinearRegression(),
    model_2=RandomForestRegressor(n_jobs=4, random_state=RANDOM_SEED),
)

hybrid.fit(X_1_train, X_2_train, y_train)

# Generate predictions and apply inverse log transformation
y_pred_train_log = hybrid.predict(X_1_train, X_2_train)
y_pred_test_log = hybrid.predict(X_1_test, X_2_test)

y_pred_train = np.expm1(y_pred_train_log)
y_pred_test = np.expm1(y_pred_test_log)
y_train_actual = np.expm1(y_train)
y_test_actual = np.expm1(y_test)

print(f"Train predictions: mean={y_pred_train.mean():.1f}, std={y_pred_train.std():.1f}")
print(f"Test predictions: mean={y_pred_test.mean():.1f}, std={y_pred_test.std():.1f}")
print(f"Train actual: mean={y_train_actual.mean():.1f}, std={y_train_actual.std():.1f}")
print(f"Test actual: mean={y_test_actual.mean():.1f}, std={y_test_actual.std():.1f}")

print("Creating future forecasts...")

# Create future dates
future_dates = pd.date_range(start=df_agg['Date'].max() + pd.Timedelta(weeks=1), 
                           periods=13, freq='W')

# Create forecast features using DeterministicProcess
X_fore = dp.out_of_sample(steps=13)
X_fore['Year'] = np.log1p(future_dates.year)
X_fore['Month'] = future_dates.month
X_fore['Week'] = future_dates.isocalendar().week

# Use last year's average for other features
forecast_features = weather_cols + ['shippingWeightKg', 'sea_distance_km']
for col in forecast_features:
    X_fore[col] = df_agg[col].mean()  # Use overall average

# Multi-step forecasting with proper lag updating
print("Generating multi-step forecasts...")

# Get last 4 actual container values for initial lags
last_containers = y.tail(4).values
print(f"Using last 4 weekly container totals for lags: {last_containers}")

# Most common HS code
most_common_hs = df_agg['hs_2digit'].mode().iloc[0]

forecast_values = []
forecast_dates = []

# Current lag values (most recent first)
current_lags = list(last_containers[::-1])

for step in range(13):
    print(f"  Step {step+1}/13...")
    
    # Create features for this step
    # X_1 should only have the deterministic + time features, not weather features
    step_X_1 = X_fore[X_1.columns].iloc[[step]].copy()
    step_X_2 = pd.DataFrame(index=[0])
    
    # Fill any NaN values in step_X_1
    step_X_1 = step_X_1.fillna(0)
    
    # Set weather and other features
    for col in forecast_features:
        step_X_2[col] = X_fore.iloc[step][col]
    
    # Set lag features (log-transformed)
    for i, lag in enumerate(['lag1', 'lag2', 'lag3', 'lag4']):
        step_X_2[lag] = np.log1p(current_lags[i]) if i < len(current_lags) else 0
    
    # Log transform other features
    step_X_2['shippingWeightKg'] = np.log1p(step_X_2['shippingWeightKg'])
    step_X_2['sea_distance_km'] = np.log1p(step_X_2['sea_distance_km'])
    step_X_2['Year'] = step_X_1['Year'].iloc[0]
    
    # Fill any NaN values in step_X_2
    step_X_2 = step_X_2.fillna(0)
    
    # One-hot encode HS code
    hs_df = pd.DataFrame({'hs_2digit': [most_common_hs]}, index=[0])
    step_X_2_cat_encoded = encoder.transform(hs_df)
    step_X_2_cat_encoded_df = pd.DataFrame(
        step_X_2_cat_encoded, 
        columns=encoded_col_names, 
        index=[0]
    )
    
    # Combine features
    step_X_2 = pd.concat([step_X_2, step_X_2_cat_encoded_df], axis=1)
    
    # Ensure all columns match training data
    for col in X_2.columns:
        if col not in step_X_2.columns:
            step_X_2[col] = 0
    
    step_X_2 = step_X_2[X_2.columns]
    
    # Generate prediction
    y_pred_step_log = hybrid.predict(step_X_1, step_X_2)
    y_pred_step = np.expm1(y_pred_step_log[0])
    
    # Store forecast
    forecast_values.append(y_pred_step)
    forecast_dates.append(future_dates[step])
    
    # Update lag features for next step
    current_lags = [y_pred_step] + current_lags[:3]

print("Creating plot...")

# Create plot
plt.figure(figsize=(16, 8))

# Prepare data for plotting
train_dates = df_agg['Date'].iloc[:split_point]
test_dates = df_agg['Date'].iloc[split_point:]

plt.plot(df_agg['Date'], y, color='gray', alpha=0.8, label='Actual', linestyle='-', linewidth=2, marker='o', markersize=3)
plt.plot(train_dates, y_pred_train, color='green', alpha=0.8, label='Train Prediction', linestyle='-', linewidth=2, marker='s', markersize=3)
plt.plot(test_dates, y_pred_test, color='blue', alpha=0.8, label='Test Prediction', linestyle='-', linewidth=2, marker='^', markersize=3)
plt.plot(forecast_dates, forecast_values, color='orange', alpha=0.8, label='Future Forecast (13 steps)', linestyle='--', linewidth=2, marker='*', markersize=5)

# Add vertical line to separate historical and forecast data
last_date = df_agg['Date'].iloc[-1]
plt.axvline(x=last_date, color='red', linestyle=':', alpha=0.7, linewidth=1, label='Forecast Start')

plt.legend()
plt.title(f"Container Import Predictions vs Actual with 13-Step Forecast\n{container_type}")
plt.ylabel("Number of Containers (Weekly Total)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary stats
print(f"\nSummary Statistics:")
print(f"Actual data points: {len(y)}")
print(f"Train prediction points: {len(y_pred_train)}")
print(f"Test prediction points: {len(y_pred_test)}")
print(f"Future forecast points: {len(forecast_values)}")

print("\nActual containers stats:")
print(y.describe())

print("\nTrain predictions stats:")
print(pd.Series(y_pred_train).describe())

print("\nTest predictions stats:")
print(pd.Series(y_pred_test).describe())

print("\nFuture forecast stats:")
print(pd.Series(forecast_values).describe())

print(f"\nForecast approach:")
print(f"- Working with aggregated weekly data throughout")
print(f"- Lag features based on actual weekly container totals") 
print(f"- Multi-step forecasting with lag updating")
print(f"- Proper inverse log transformation applied")
print(f"- HS Code used: {most_common_hs} (most common for this container type)")
print(f"- Forecast period: {forecast_dates[0].strftime('%Y-%m-%d')} to {forecast_dates[-1].strftime('%Y-%m-%d')}")

print("\nForecast dates:")
for i, (date, pred) in enumerate(zip(forecast_dates, forecast_values)):
    print(f"  Step {i+1}: {date.strftime('%Y-%m-%d')} - {pred:.0f} containers") 