"""Entraîne et sauvegarde le modèle : python -m wine_quality.train"""
from .model import entrainer


def main():
    m = entrainer()["metrics"]
    print(f"Random Forest, test ({m['n_test']} vins) : accuracy={m['accuracy']:.3f}  precision={m['precision']:.3f}  "
          f"rappel={m['recall']:.3f}  F1={m['f1']:.3f}  AUC={m['auc']:.3f}")
    print("Modèle sauvegardé dans le dossier models/")


if __name__ == "__main__":
    main()
