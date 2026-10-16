import pytest

from app import create_app
from wine_quality.config import EXEMPLES


@pytest.fixture
def client():
    return create_app().test_client()


def formulaire(exemple):
    return {k.replace(" ", "_"): v for k, v in EXEMPLES[exemple].items()}


def test_page_d_accueil(client):
    r = client.get("/")
    assert r.status_code == 200
    texte = r.get_data(as_text=True)
    assert "Ce vin est-il bon" in texte
    assert texte.count('type="number"') == 11


def test_sante(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_formulaire_bon_vin(client):
    r = client.post("/", data=formulaire("bon"))
    assert r.status_code == 200
    texte = r.get_data(as_text=True)
    assert "Bon vin" in texte and "Pas un bon vin" not in texte


def test_formulaire_vin_ordinaire(client):
    r = client.post("/", data=formulaire("ordinaire"))
    assert r.status_code == 200
    assert "Pas un bon vin" in r.get_data(as_text=True)


def test_formulaire_invalide(client):
    r = client.post("/", data={"alcohol": "abc"})
    assert r.status_code == 400
    assert "Saisie incorrecte" in r.get_data(as_text=True)


def test_api_json(client):
    r = client.post("/api/predict", json={"valeurs": EXEMPLES["bon"]})
    assert r.status_code == 200
    assert r.get_json()["label"] == "bon"


def test_api_valeurs_manquantes(client):
    r = client.post("/api/predict", json={"valeurs": {}})
    assert r.status_code == 400
    assert "erreur" in r.get_json()


def test_api_sans_corps(client):
    assert client.post("/api/predict").status_code == 400
