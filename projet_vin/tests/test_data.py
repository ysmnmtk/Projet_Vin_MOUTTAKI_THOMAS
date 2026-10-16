import pytest

from wine_quality import data
from wine_quality.config import FEATURES


def test_chargement_fusionne_rouge_et_blanc():
    df = data.charger_brut()
    assert df.shape == (6497, 13)
    assert df["couleur"].value_counts().to_dict() == {"blanc": 4898, "rouge": 1599}


def test_nettoyage_supprime_les_doublons():
    df = data.charger_propre()
    assert len(df) == 5320
    assert not df.duplicated().any()


def test_toutes_les_variables_sont_gardees():
    assert len(FEATURES) == 11
    assert set(FEATURES) <= set(data.charger_propre().columns)


def test_cible_binaire_sur_petit_exemple():
    import pandas as pd
    df = pd.DataFrame({"quality": [5, 6, 7, 8], "alcohol": [9, 10, 11, 12]})
    assert data.nettoyer(df)["good"].tolist() == [0, 0, 1, 1]


def test_classes_desequilibrees():
    df = data.charger_propre()
    assert df["good"].mean() == pytest.approx(0.19, abs=0.005)
    assert df[df["couleur"] == "rouge"]["good"].mean() < df[df["couleur"] == "blanc"]["good"].mean()


def test_separation_stratifiee_sans_fuite():
    X_train11, X_test11, y_train, y_test = data.separer(data.charger_propre())
    assert (len(y_train), len(y_test)) == (4256, 1064)
    assert ((y_train == 0).sum(), (y_train == 1).sum()) == (3449, 807)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert set(X_train11.index).isdisjoint(X_test11.index)
    assert list(X_train11.columns) == FEATURES


def test_separation_reproductible():
    df = data.charger_propre()
    assert list(data.separer(df)[1].index) == list(data.separer(df)[1].index)


def test_plages():
    p = data.plages()
    assert set(p) == set(FEATURES)
    assert all(mini <= maxi for mini, maxi in p.values())
