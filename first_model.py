from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

# 1. Load the training data
df_train = pd.read_parquet("../input/mdsb-2023/train.parquet")
df_test = pd.read_parquet("../input/mdsb-2023/test.parquet")
#df_train = pd.read_parquet(Path("data") / "train.parquet")
#df_test = pd.read_parquet(Path("data") / "final_test.parquet")

# 2. Define feature engineering functions
def _encode_dates(X):
    X = X.copy()
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X["is_weekend"] = X["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    return X

def encode_cyclical_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def calculate_distance_to_center(df):
    city_center = (48.8530, 2.3499)
    df['distance_to_center'] = np.sqrt(
        (df['latitude'] - city_center[0])**2 + (df['longitude'] - city_center[1])**2
    )
    return df

# 3. Preprocess the training data
df_train_fe = df_train.copy()
df_train_fe = _encode_dates(df_train_fe)
df_train_fe = encode_cyclical_features(df_train_fe)
df_train_fe = calculate_distance_to_center(df_train_fe)
df_train_fe = df_train_fe.sort_values(['counter_name', 'date'])
df_train_fe['bike_count_lag_1'] = df_train_fe.groupby('counter_name')['bike_count'].shift(1)
df_train_fe['bike_count_roll_mean_3'] = df_train_fe.groupby('counter_name')['bike_count'].rolling(window=3).mean().reset_index(level=0, drop=True)
df_train_fe['bike_count_roll_mean_168'] = df_train_fe.groupby('counter_name')['bike_count'].rolling(window=168).mean().reset_index(level=0, drop=True)
df_train_fe = df_train_fe.dropna(subset=['bike_count_lag_1', 'bike_count_roll_mean_3', 'bike_count_roll_mean_168'])
df_train_fe['log_bike_count'] = df_train_fe['log_bike_count']

# 4. Define features and target
features = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_weekend', 'distance_to_center',
    'bike_count_lag_1', 'bike_count_roll_mean_3', 'bike_count_roll_mean_168',
    'counter_name', 'site_name'
]
X_train = df_train_fe[features]
y_train = df_train_fe['log_bike_count']

# 5. Preprocessing pipelines
numeric_features = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_weekend', 'distance_to_center',
    'bike_count_lag_1', 'bike_count_roll_mean_3', 'bike_count_roll_mean_168'
]
categorical_features = ['counter_name', 'site_name']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 6. Define and train the model
model = Ridge()
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])
pipeline.fit(X_train, y_train)

# 7. Load and preprocess the test data
df_test_fe = df_test.copy()
df_test_fe = _encode_dates(df_test_fe)
df_test_fe = encode_cyclical_features(df_test_fe)
df_test_fe = calculate_distance_to_center(df_test_fe)

# Combine train and test data for lag features
df_combined = pd.concat([df_train_fe, df_test_fe], sort=False)
df_combined = df_combined.sort_values(['counter_name', 'date'])
df_combined['bike_count_lag_1'] = df_combined.groupby('counter_name')['bike_count'].shift(1)
df_combined['bike_count_roll_mean_3'] = df_combined.groupby('counter_name')['bike_count'].rolling(window=3).mean().reset_index(level=0, drop=True)
df_combined['bike_count_roll_mean_168'] = df_combined.groupby('counter_name')['bike_count'].rolling(window=168).mean().reset_index(level=0, drop=True)

# Extract test features
df_test_fe = df_combined[df_combined['date'].isin(df_test_fe['date'])]

# Handle missing values in lag features
df_test_fe['bike_count_lag_1'] = df_test_fe['bike_count_lag_1'].fillna(df_train_fe['bike_count_lag_1'].mean())
df_test_fe['bike_count_roll_mean_3'] = df_test_fe['bike_count_roll_mean_3'].fillna(df_train_fe['bike_count_roll_mean_3'].mean())
df_test_fe['bike_count_roll_mean_168'] = df_test_fe['bike_count_roll_mean_168'].fillna(df_train_fe['bike_count_roll_mean_168'].mean())

X_test = df_test_fe[features]

# 8. Make predictions
y_pred = pipeline.predict(X_test)

# 9. Prepare submission file
results = pd.DataFrame({
    'Id': np.arange(y_pred.shape[0]),
    'log_bike_count': y_pred
})
results.to_csv("submission.csv", index=False)
