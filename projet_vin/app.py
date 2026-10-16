import os

from flask import Flask, jsonify, render_template, request

from wine_quality.config import EXEMPLES, FEATURES, LABELS
from wine_quality.data import plages
from wine_quality.predict import predire


def slug(nom):
    return nom.replace(" ", "_")


def create_app():
    app = Flask(__name__)

    def contexte(**extra):
        champs = [{"nom": f, "slug": slug(f), "label": LABELS[f]} for f in FEATURES]
        base = {"champs": champs, "valeurs": {}, "resultat": None, "erreur": None,
                "exemples": EXEMPLES, "plages": plages()}
        base.update(extra)
        return render_template("index.html", **base)

    @app.get("/")
    def accueil():
        return contexte()

    @app.post("/")
    def formulaire():
        valeurs = {f: request.form.get(slug(f), "") for f in FEATURES}
        try:
            return contexte(valeurs=valeurs, resultat=predire(valeurs))
        except ValueError as e:
            return contexte(valeurs=valeurs, erreur=str(e)), 400

    @app.post("/api/predict")
    def api_predict():
        donnees = request.get_json(silent=True) or {}
        valeurs = {str(k).replace("_", " "): v for k, v in (donnees.get("valeurs") or {}).items()}
        try:
            return jsonify(predire(valeurs))
        except ValueError as e:
            return jsonify({"erreur": str(e)}), 400

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)))
