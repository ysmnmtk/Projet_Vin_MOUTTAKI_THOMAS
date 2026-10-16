# Classification de la qualité du vin

Projet de Machine Learning 1 (EFREI, 2026-2027), réalisé par Soline Thomas et Yasmine Mouttaki.

On prédit si un vin est bon (note d'au moins 7) à partir de 11 mesures chimiques. Cinq modèles sont comparés : régression logistique, k-NN, arbre de décision, SVM et Random Forest. La Random Forest est la meilleure (F1 de 0,611 sur le test) et c'est elle qu'utilise l'application web.

## Contenu du dossier

- `projet_vin.ipynb` : le notebook avec tout le code et les explications.
- `data/` : les fichiers de données (vins rouges et blancs).
- `wine_quality/` : le code de l'application (chargement des données, modèle, prédiction).
- `app.py` et `templates/` : l'application web.
- `tests/` : les tests.
- `Dockerfile` et `docker-compose.yml` : pour lancer le projet avec Docker.

## Lancer le projet

Installer les paquets :

```
pip install -r requirements.txt
```

Ouvrir le notebook :

```
jupyter notebook
```

Lancer l'application web, puis ouvrir http://127.0.0.1:5000 :

```
python app.py
```

Lancer les tests :

```
pytest
```

Lancer avec Docker :

```
docker compose up --build
```

