from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from lightgbm import LGBMRegressor


def _encode_dates(X):
    X = X.copy()
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday
    X["hour"] = X["date"].dt.hour
    X["week_of_year"] = X["date"].dt.isocalendar().week
    X["is_weekend"] = (X["weekday"] >= 5).astype(int)  # Ajout d'un indicateur weekend
    return X.drop(columns=["date"])


def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data/external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    X["date"] = pd.to_datetime(X["date"], errors="coerce").astype("datetime64[ns]")
    df_ext["date"] = pd.to_datetime(df_ext["date"], errors="coerce").astype("datetime64[ns]")

    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"),
        df_ext[["date", "t", "rr1", "ff"]].sort_values("date"),
        on="date",
    )
    X = X.sort_values("orig_index").drop(columns=["orig_index"])
    return X


def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour", "week_of_year", "is_weekend"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
            ("num", StandardScaler(), ["t", "rr1", "ff"]),
        ]
    )

    # Optimisation des hyperparamètres LightGBM
    regressor = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.02,  # Réduction du learning rate pour une meilleure convergence
        max_depth=10,
        min_child_weight=1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.5,  # Régularisation L1 plus forte
        reg_lambda=2.5,  # Régularisation L2 plus forte
        num_leaves=31,
        random_state=42,
    )

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe


# Charger les données
train_path = "input/mdsb-2023/train.parquet"
df_train = pd.read_parquet(train_path)

# Définir X et y
X_train = df_train.drop(columns=["bike_count"])
y_train = np.log1p(df_train["bike_count"])

# Validation croisée avec TimeSeriesSplit
model = get_estimator()
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="neg_root_mean_squared_error")
print(f"Cross-validated RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")

# Entraîner le modèle
model.fit(X_train, y_train)

# Charger les données de test
test_path = "input/mdsb-2023/final_test.parquet"
df_test = pd.read_parquet(test_path)

# Prédictions
y_pred = model.predict(df_test)

# Générer le fichier de soumission
results = pd.DataFrame({"Id": np.arange(len(y_pred)), "log_bike_count": y_pred})
results.to_csv("submission_v6.csv", index=False)
print("Fichier de soumission créé : submission_v6.csv")
