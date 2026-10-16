import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .config import FEATURES, MODEL_DIR, RF_PARAMS
from .data import charger_propre, separer

_cache = {}


def creer_modele():
    return RandomForestClassifier(**RF_PARAMS, random_state=42, n_jobs=-1)


def evaluer():
    """Entraîne sur 80 % des vins et note sur les 20 % restants (jamais vus ni rééquilibrés)."""
    X_train11, X_test11, y_train, y_test = separer(charger_propre())
    modele = creer_modele().fit(X_train11, y_train)
    pred = modele.predict(X_test11)
    proba = modele.predict_proba(X_test11)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "auc": roc_auc_score(y_test, proba),
        "n_test": int(len(y_test)),
    }


def entrainer(sauvegarder=True):
    """Évalue le modèle, puis le ré-entraîne sur tous les vins pour l'utilisation réelle."""
    metriques = evaluer()
    df = charger_propre()
    modele = creer_modele().fit(df[FEATURES], df["good"])
    bundle = {"model": modele, "features": FEATURES, "metrics": metriques, "sklearn": sklearn.__version__}
    if sauvegarder:
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, MODEL_DIR / "model.joblib")
    _cache["bundle"] = bundle
    return bundle


def charger():
    """Charge le modèle sauvegardé ; s'il manque ou vient d'une autre version de scikit-learn, on le ré-entraîne."""
    if "bundle" in _cache:
        return _cache["bundle"]
    try:
        bundle = joblib.load(MODEL_DIR / "model.joblib")
        if bundle.get("sklearn") != sklearn.__version__:
            raise ValueError("version de scikit-learn différente")
        _cache["bundle"] = bundle
        return bundle
    except Exception:
        return entrainer()
