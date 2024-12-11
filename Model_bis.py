import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

# Charger les données
df_train = pd.read_parquet("../input/mdsb-2023/train.parquet")
df_test = pd.read_parquet("../input/mdsb-2023/test.parquet")

# Fonction d'extraction des dates
def _encode_dates(X):
    X = X.copy()
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X["is_weekend"] = X["weekday"].apply(lambda x: 1 if x >= 5 else 0)
    return X

# Fonction de transformation cyclique
def encode_cyclical_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

# Fonction de calcul de distance au centre-ville
def calculate_distance_to_center(df):
    city_center = (48.8530, 2.3499)
    df['distance_to_center'] = np.sqrt(
        (df['latitude'] - city_center[0])**2 + (df['longitude'] - city_center[1])**2
    )
    return df

# Fonction pour créer des lag features
def create_lag_features(df):
    df = df.sort_values(['counter_name', 'date'])
    df['bike_count_lag_1'] = df.groupby('counter_name')['bike_count'].shift(1)
    df['bike_count_roll_mean_3'] = df.groupby('counter_name')['bike_count'].rolling(window=3).mean().reset_index(level=0, drop=True)
    df['bike_count_roll_mean_168'] = df.groupby('counter_name')['bike_count'].rolling(window=168).mean().reset_index(level=0, drop=True)
    return df

# Copie des données
df_train_fe = df_train.copy()

# Feature Engineering
df_train_fe = _encode_dates(df_train_fe)
df_train_fe = encode_cyclical_features(df_train_fe)
df_train_fe = calculate_distance_to_center(df_train_fe)
df_train_fe = create_lag_features(df_train_fe)

# Supprimer les lignes avec des NaN
df_train_fe = df_train_fe.dropna(subset=['bike_count_lag_1', 'bike_count_roll_mean_3', 'bike_count_roll_mean_168'])

# Transformation cible (logarithmique)
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

# Initialiser LightGBM
model = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=10,
    random_state=42
)

# Entraîner le modèle
model.fit(X_train, y_train)

# Validation croisée avec TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_root_mean_squared_error')
print(f"RMSE moyen : {-scores.mean():.2f}")

# Préparer les données de test
df_test_fe = df_test.copy()
df_test_fe = _encode_dates(df_test_fe)
df_test_fe = encode_cyclical_features(df_test_fe)
df_test_fe = calculate_distance_to_center(df_test_fe)
df_test_fe = create_lag_features(df_test_fe)

# Remplir les valeurs manquantes
df_test_fe['bike_count_lag_1'] = df_test_fe['bike_count_lag_1'].fillna(df_train_fe['bike_count_lag_1'].mean())
df_test_fe['bike_count_roll_mean_3'] = df_test_fe['bike_count_roll_mean_3'].fillna(df_train_fe['bike_count_roll_mean_3'].mean())
df_test_fe['bike_count_roll_mean_168'] = df_test_fe['bike_count_roll_mean_168'].fillna(df_train_fe['bike_count_roll_mean_168'].mean())

# Extraire les features
X_test = df_test_fe[features]

# Appliquer l'encodage des catégories
for col in ['counter_name', 'site_name']:
    X_test[col] = label_encoders[col].transform(X_test[col])

# Prédire les résultats
y_pred = model.predict(X_test)

# Créer le fichier de soumission
results = pd.DataFrame({
    'Id': np.arange(len(y_pred)),
    'log_bike_count': y_pred
})
results.to_csv("submission.csv", index=False)

# Analyse des erreurs
residuals = y_train - model.predict(X_train)
import matplotlib.pyplot as plt
plt.scatter(y_train, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.title('Residual Analysis')
plt.show()

# Optimisation des hyperparamètres avec GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [6, 8, 10],
}

grid = GridSearchCV(
    estimator=LGBMRegressor(random_state=42),
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error'
)
grid.fit(X_train, y_train)
print(f"Meilleurs paramètres : {grid.best_params_}")
