from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

FILES = {"rouge": "winequality-red.csv", "blanc": "winequality-white.csv"}

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
]

RF_PARAMS = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5,
             "max_features": "sqrt", "class_weight": "balanced"}

SEUIL_QUALITE = 7
SEUIL_BON = 0.5

LABELS = {
    "fixed acidity": "Acidité fixe (g/L)",
    "volatile acidity": "Acidité volatile (g/L)",
    "citric acid": "Acide citrique (g/L)",
    "residual sugar": "Sucre résiduel (g/L)",
    "chlorides": "Chlorures (g/L)",
    "free sulfur dioxide": "SO₂ libre (mg/L)",
    "total sulfur dioxide": "SO₂ total (mg/L)",
    "density": "Densité (g/cm³)",
    "pH": "pH",
    "sulphates": "Sulfates (g/L)",
    "alcohol": "Alcool (% vol.)",
}

EXEMPLES = {
    "bon": {"fixed acidity": 5.0, "volatile acidity": 0.455, "citric acid": 0.18, "residual sugar": 1.9,
            "chlorides": 0.036, "free sulfur dioxide": 33.0, "total sulfur dioxide": 106.0,
            "density": 0.98746, "pH": 3.21, "sulphates": 0.83, "alcohol": 14.0},
    "ordinaire": {"fixed acidity": 7.7, "volatile acidity": 0.64, "citric acid": 0.21, "residual sugar": 2.2,
                  "chlorides": 0.077, "free sulfur dioxide": 32.0, "total sulfur dioxide": 133.0,
                  "density": 0.9956, "pH": 3.27, "sulphates": 0.45, "alcohol": 9.9},
}
