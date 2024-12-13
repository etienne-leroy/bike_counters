from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour
    return X.drop(columns=["date"])


def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data/external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    X["date"] = X["date"].astype("datetime64[ns]")  # Convertir en nanosecondes
    df_ext["date"] = df_ext["date"].astype("datetime64[ns]")  # Convertir en nanosecondes

    # Ajouter un index temporaire pour trier
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"),
        df_ext[["date", "t", "rr1", "ff"]].sort_values("date"),  # Colonnes météo
        on="date",
    )
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X



def get_estimator():
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
            ("num", "passthrough", ["t", "rr1", "ff"]),  # Intégrer les variables météo
        ]
    )

    regressor = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=10, random_state=42
    )

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe


# Charger les données d'entraînement
train_path = "input/mdsb-2023/train.parquet"
df_train = pd.read_parquet(train_path)

# Définir X et y
X_train = df_train.drop(columns=["bike_count"])
y_train = np.log1p(df_train["bike_count"])  # Transformation logarithmique

# Créer et entraîner le pipeline
model = get_estimator()
model.fit(X_train, y_train)

# Charger et prédire sur les données de test
test_path = "input/mdsb-2023/final_test.parquet"
df_test = pd.read_parquet(test_path)
y_pred = model.predict(df_test)

# Générer le fichier de soumission
results = pd.DataFrame({"Id": np.arange(len(y_pred)), "log_bike_count": y_pred})
results.to_csv("submission_v2.csv", index=False)



