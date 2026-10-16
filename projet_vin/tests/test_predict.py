import pytest

from wine_quality.config import EXEMPLES
from wine_quality.predict import predire


def test_exemple_bon_vin():
    r = predire(EXEMPLES["bon"])
    assert r["label"] == "bon"
    assert r["score"] >= 0.5


def test_exemple_vin_ordinaire():
    r = predire(EXEMPLES["ordinaire"])
    assert r["label"] == "pas bon"
    assert 0 <= r["score"] < 0.5


def test_valeur_manquante():
    valeurs = dict(EXEMPLES["bon"])
    del valeurs["alcohol"]
    with pytest.raises(ValueError, match="alcohol"):
        predire(valeurs)


@pytest.mark.parametrize("mauvaise", ["abc", "-3", "nan", "inf", ""])
def test_valeur_invalide(mauvaise):
    with pytest.raises(ValueError):
        predire(dict(EXEMPLES["bon"], alcohol=mauvaise))


def test_virgule_decimale_acceptee():
    assert predire(dict(EXEMPLES["bon"], **{"volatile acidity": "0,455"}))["label"] == "bon"


def test_avertissement_hors_plage():
    assert predire(dict(EXEMPLES["bon"], alcohol=40))["avertissements"]
    assert predire(EXEMPLES["bon"])["avertissements"] == []
