import math

import pandas as pd

from .config import SEUIL_BON
from .data import plages
from .model import charger


def _nombre(valeur, nom):
    if valeur is None or str(valeur).strip() == "":
        raise ValueError(f"Valeur manquante : {nom}")
    try:
        x = float(str(valeur).replace(",", "."))
    except ValueError:
        raise ValueError(f"Valeur non numérique pour {nom} : {valeur!r}")
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"Valeur invalide pour {nom} : {valeur!r} (un nombre positif est attendu)")
    return x


def predire(valeurs):
    """Prédit si un vin est bon. `valeurs` : dictionnaire {nom de variable: valeur} (les 11 mesures)."""
    bundle = charger()
    colonnes = bundle["features"]
    ligne = {c: _nombre(valeurs.get(c), c) for c in colonnes}

    limites = plages()
    avertissements = [
        f"{c} = {ligne[c]:g} est en dehors des valeurs observées ({limites[c][0]:g} à {limites[c][1]:g}) : la prédiction est moins fiable."
        for c in colonnes if not limites[c][0] <= ligne[c] <= limites[c][1]
    ]

    X = pd.DataFrame([ligne], columns=colonnes)
    score = float(bundle["model"].predict_proba(X)[0, 1])
    return {
        "label": "bon" if score >= SEUIL_BON else "pas bon",
        "score": score,
        "avertissements": avertissements,
        "metriques": bundle["metrics"],
    }
