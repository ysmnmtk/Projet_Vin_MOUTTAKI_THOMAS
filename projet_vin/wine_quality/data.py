import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, FEATURES, FILES, SEUIL_QUALITE


def charger_brut():
    """Charge les vins rouges et blancs et les met ensemble (colonne `couleur`)."""
    morceaux = []
    for couleur, fichier in FILES.items():
        df = pd.read_csv(DATA_DIR / fichier, sep=";")
        df["couleur"] = couleur
        morceaux.append(df)
    return pd.concat(morceaux, ignore_index=True)


def nettoyer(df):
    """Supprime les doublons et ajoute la cible binaire `good` (qualité >= 7)."""
    out = df.drop_duplicates().reset_index(drop=True)
    out["good"] = (out["quality"] >= SEUIL_QUALITE).astype(int)
    return out


def charger_propre():
    return nettoyer(charger_brut())


def separer(df, test_size=0.2, seed=42):
    """Sépare en train et test (stratifié). Le test n'est jamais rééquilibré."""
    return train_test_split(df[FEATURES], df["good"], test_size=test_size, stratify=df["good"], random_state=seed)


def plages():
    """Minimum et maximum observés de chaque variable (pour signaler les valeurs inhabituelles)."""
    df = charger_propre()
    return {c: (float(df[c].min()), float(df[c].max())) for c in FEATURES}
