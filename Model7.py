from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor

# Fonction pour encoder les dates
def _encode_dates(X):
    X = X.copy()
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    return X.drop(columns=["date"])

# Fonction pour ajouter des lag features
def _add_lag_features(X):
    X = X.copy()
    if "bike_count" in X.columns:
        X["bike_count_lag_1"] = X["bike_count"].shift(1)
        X["bike_count_lag_7"] = X["bike_count"].shift(7)
        X["bike_count_roll_mean_7"] = X["bike_count"].rolling(7).mean()
        X["bike_count_roll_mean_14"] = X["bike_count"].rolling(14).mean()
    else:
        # Si 'bike_count' n'existe pas, initialiser les colonnes à 0
        X["bike_count_lag_1"] = 0
        X["bike_count_lag_7"] = 0
        X["bike_count_roll_mean_7"] = 0
        X["bike_count_roll_mean_14"] = 0
    return X

# Fonction pour intégrer les données externes
def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data/external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])
    X = X.copy()
    X["date"] = X["date"].astype("datetime64[ns]")
    df_ext["date"] = df_ext["date"].astype("datetime64[ns]")

    # Ajouter un index temporaire pour trier
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"),
        df_ext[["date", "t", "rr1", "ff"]].sort_values("date"),
        on="date",
    )
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

# Création du modèle
def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    lag_feature_adder = FunctionTransformer(_add_lag_features, validate=False)

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("cat", categorical_encoder, categorical_cols),
            ("num", "passthrough", ["t", "rr1", "ff", "bike_count_lag_1", "bike_count_lag_7", 
                                    "bike_count_roll_mean_7", "bike_count_roll_mean_14"]),
        ]
    )

    regressor = LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=12, random_state=42
    )

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        lag_feature_adder,
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe

# Chargement des données
train_path = "input/mdsb-2023/train.parquet"
df_train = pd.read_parquet(train_path)

# Préparer les données d'entraînement
df_train = _add_lag_features(df_train)  # Ajout des lag features sur l'ensemble du jeu d'entraînement
X_train = df_train.drop(columns=["bike_count"])
y_train = np.log1p(df_train["bike_count"])

# Créer et entraîner le modèle
model = get_estimator()
model.fit(X_train, y_train)

# Chargement des données de test
test_path = "input/mdsb-2023/final_test.parquet"
df_test = pd.read_parquet(test_path)
df_test = _add_lag_features(df_test)

# Prédictions
y_pred = model.predict(df_test)

# Générer le fichier de soumission
results = pd.DataFrame({"Id": np.arange(len(y_pred)), "log_bike_count": y_pred})
results.to_csv("submission_v5.csv", index=False)
