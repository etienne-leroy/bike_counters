import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from lightgbm import LGBMRegressor

# Charger les données
df_train = pd.read_parquet("input/mdsb-2023/train.parquet")
df_test = pd.read_parquet("input/mdsb-2023/final_test.parquet")


# 1. Définir les fonctions de prétraitement
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

def create_lag_features(df):
    df = df.sort_values(['counter_name', 'date'])
    df['bike_count_lag_1'] = df.groupby('counter_name')['bike_count'].shift(1)
    df['bike_count_roll_mean_3'] = df.groupby('counter_name')['bike_count'].rolling(window=3).mean().reset_index(level=0, drop=True)
    df['bike_count_roll_mean_168'] = df.groupby('counter_name')['bike_count'].rolling(window=168).mean().reset_index(level=0, drop=True)
    return df

# 2. Prétraitement des données d'entraînement
df_train_fe = df_train.copy()
df_train_fe = _encode_dates(df_train_fe)
df_train_fe = encode_cyclical_features(df_train_fe)
df_train_fe = calculate_distance_to_center(df_train_fe)
df_train_fe = create_lag_features(df_train_fe)
df_train_fe = df_train_fe.dropna(subset=['bike_count_lag_1', 'bike_count_roll_mean_3', 'bike_count_roll_mean_168'])
df_train_fe['log_bike_count'] = np.log1p(df_train_fe['bike_count'])

# Définir les features et la cible
features = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_weekend', 'distance_to_center',
    'bike_count_lag_1', 'bike_count_roll_mean_3', 'bike_count_roll_mean_168',
    'counter_name', 'site_name'
]
X_train = df_train_fe[features]
y_train = df_train_fe['log_bike_count']

# Encodage des catégories
label_encoders = {}
for col in ['counter_name', 'site_name']:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    label_encoders[col] = le

# 3. Définir et entraîner le modèle
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# Validation croisée avec TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error')
print(f"RMSE moyen : {-scores.mean():.2f}")

# 4. Prétraitement des données de test
df_test_fe = df_test.copy()
df_test_fe = _encode_dates(df_test_fe)
df_test_fe = encode_cyclical_features(df_test_fe)
df_test_fe = calculate_distance_to_center(df_test_fe)

# Combiner les données d'entraînement et de test pour les lag features
df_combined = pd.concat([df_train_fe, df_test_fe], sort=False)
df_combined = df_combined.sort_values(['counter_name', 'date'])
df_combined = create_lag_features(df_combined)

# Extraire les features pour les données de test
df_test_fe = df_combined[df_combined['date'].isin(df_test_fe['date'])]
df_test_fe['bike_count_lag_1'] = df_test_fe['bike_count_lag_1'].fillna(df_train_fe['bike_count_lag_1'].mean())
df_test_fe['bike_count_roll_mean_3'] = df_test_fe['bike_count_roll_mean_3'].fillna(df_train_fe['bike_count_roll_mean_3'].mean())
df_test_fe['bike_count_roll_mean_168'] = df_test_fe['bike_count_roll_mean_168'].fillna(df_train_fe['bike_count_roll_mean_168'].mean())

X_test = df_test_fe[features]

# Appliquer l'encodage des catégories pour les données de test
for col in ['counter_name', 'site_name']:
    X_test[col] = label_encoders[col].transform(X_test[col])

# 5. Prédictions et génération du fichier de soumission
y_pred = model.predict(X_test)
results = pd.DataFrame({
    'Id': np.arange(len(y_pred)),
    'log_bike_count': y_pred
})
results.to_csv("submission.csv", index=False)
print("Le fichier 'submission.csv' a été généré pour Kaggle.")
