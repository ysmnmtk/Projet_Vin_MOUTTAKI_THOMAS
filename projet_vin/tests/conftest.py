import pytest


@pytest.fixture(scope="session", autouse=True)
def dossier_modeles(tmp_path_factory):
    """Les tests sauvegardent le modèle dans un dossier temporaire, pas dans le projet."""
    mp = pytest.MonkeyPatch()
    mp.setattr("wine_quality.model.MODEL_DIR", tmp_path_factory.mktemp("models"))
    yield
    mp.undo()
