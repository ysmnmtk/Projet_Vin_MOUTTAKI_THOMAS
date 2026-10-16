import joblib
import sklearn

from wine_quality import model
from wine_quality.config import FEATURES, RF_PARAMS


def test_modele_utilise_class_weight_balanced_et_les_11_variables():
    m = model.creer_modele()
    assert m.get_params()["class_weight"] == "balanced"
    assert m.get_params()["n_estimators"] == RF_PARAMS["n_estimators"]
    assert model.entrainer(sauvegarder=False)["features"] == FEATURES


def test_evaluation_bat_la_baseline():
    m = model.evaluer()
    assert set(m) == {"accuracy", "precision", "recall", "f1", "auc", "n_test"}
    assert m["n_test"] == 1064
    assert m["auc"] > 0.85
    assert m["f1"] > 0.55
    assert m["recall"] > 0.60


def test_charger_entraine_si_le_modele_manque(monkeypatch, tmp_path):
    monkeypatch.setattr(model, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(model, "_cache", {})
    bundle = model.charger()
    assert (tmp_path / "model.joblib").exists()
    assert bundle["features"] == FEATURES


def test_charger_reentraine_si_version_differente(monkeypatch, tmp_path):
    monkeypatch.setattr(model, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(model, "_cache", {})
    joblib.dump({"model": None, "features": [], "metrics": {}, "sklearn": "0.0.0"}, tmp_path / "model.joblib")
    bundle = model.charger()
    assert bundle["sklearn"] == sklearn.__version__
    assert bundle["model"] is not None
