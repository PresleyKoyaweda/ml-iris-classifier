"""
Interface Streamlit pour consommer le endpoint Azure ML Managed Online
Endpoint qui héberge le modèle Iris (déployé via scripts/deploy_endpoint.py
et terraform/main.tf).

Deux modes:
    - Prédiction unique: caractéristiques saisies manuellement.
    - Prédiction par fichier: import d'un CSV ou Excel pour prédire
      plusieurs fleurs d'un coup.

Lancement local:
    streamlit run src/webapp/app.py

Configuration (variables d'environnement ou fichier .env à la racine):
    AZURE_ML_ENDPOINT_URL   URL de scoring de l'endpoint (scoring_uri)
    AZURE_ML_ENDPOINT_KEY   Clé d'authentification de l'endpoint
"""

import os
import re
import unicodedata

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ordre des classes retourné par score.py: model.predict() renvoie l'index
# tel qu'encodé par sklearn.datasets.load_iris (0=setosa, 1=versicolor,
# 2=virginica) — voir src/train/train.py.
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Ordre attendu par score.py pour la clé "data" (voir src/train/outputs/metadata.json).
FEATURE_ORDER = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

SAMPLES = {
    "Setosa (exemple)": (5.1, 3.5, 1.4, 0.2),
    "Versicolor (exemple)": (6.0, 2.7, 4.2, 1.3),
    "Virginica (exemple)": (6.9, 3.1, 5.4, 2.1),
}

# Reconnaît les colonnes d'un fichier importé quel que soit leur libellé exact
# (anglais/français, avec ou sans unité) tant qu'elles contiennent ces mots-clés.
FEATURE_MATCHERS = {
    "sepal_length": lambda n: "sepal" in n and ("length" in n or "longueur" in n),
    "sepal_width": lambda n: "sepal" in n and ("width" in n or "largeur" in n),
    "petal_length": lambda n: "petal" in n and ("length" in n or "longueur" in n),
    "petal_width": lambda n: "petal" in n and ("width" in n or "largeur" in n),
}

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

st.title("🌸 Iris Classifier")
st.caption("Interface web pour le modèle hébergé sur Azure ML Managed Online Endpoint")

endpoint_url = os.getenv("AZURE_ML_ENDPOINT_URL", "")
endpoint_key = os.getenv("AZURE_ML_ENDPOINT_KEY", "")

with st.sidebar:
    st.header("⚙️ Configuration")
    endpoint_url = st.text_input("Endpoint URL", value=endpoint_url, help="scoring_uri de l'endpoint Azure ML")
    endpoint_key = st.text_input("Clé d'API", value=endpoint_key, type="password")

    if not endpoint_url or not endpoint_key:
        st.warning(
            "Renseigne AZURE_ML_ENDPOINT_URL et AZURE_ML_ENDPOINT_KEY "
            "(variables d'environnement, fichier .env, ou champs ci-dessus)."
        )

    st.divider()
    st.caption(
        "Récupère ces valeurs avec:\n\n"
        "`az ml online-endpoint show -n iris-classifier "
        "--query scoring_uri`\n\n"
        "`az ml online-endpoint get-credentials -n iris-classifier "
        "--query primaryKey`"
    )


def call_endpoint(features: list) -> dict:
    """Appelle le endpoint de scoring pour une fleur et renvoie {prediction, probabilities}."""
    response = requests.post(
        endpoint_url,
        json={"data": [features]},
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {endpoint_key}",
        },
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    if "error" in result:
        raise ValueError(result["error"])
    return result


def normalize_header(name: str) -> str:
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def map_columns(columns) -> dict:
    """Associe les colonnes du fichier importé aux 4 features attendues.

    Lève ValueError si une feature est introuvable ou ambiguë (plusieurs
    colonnes correspondent), avec un message actionnable pour l'utilisateur.
    """
    normalized = {col: normalize_header(col) for col in columns}
    mapping = {}
    for feature, matches in FEATURE_MATCHERS.items():
        found = [col for col, n in normalized.items() if matches(n)]
        if not found:
            raise ValueError(
                f"Colonne manquante pour '{feature}' "
                f"(ex: sepal_length, 'sepal length (cm)', longueur_sepale...)."
            )
        if len(found) > 1:
            raise ValueError(f"Plusieurs colonnes correspondent à '{feature}': {found}")
        mapping[feature] = found[0]
    return mapping


tab_single, tab_batch = st.tabs(["🔮 Prédiction unique", "📁 Prédiction par fichier"])

with tab_single:
    st.subheader("Caractéristiques de la fleur")

    preset = st.selectbox("Charger un exemple", ["—"] + list(SAMPLES.keys()))
    defaults = SAMPLES.get(preset, (5.1, 3.5, 1.4, 0.2))

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider("Longueur sépale (cm)", 4.0, 8.0, defaults[0], 0.1)
        petal_length = st.slider("Longueur pétale (cm)", 1.0, 7.0, defaults[2], 0.1)
    with col2:
        sepal_width = st.slider("Largeur sépale (cm)", 2.0, 4.5, defaults[1], 0.1)
        petal_width = st.slider("Largeur pétale (cm)", 0.1, 2.5, defaults[3], 0.1)

    predict_clicked = st.button("🔮 Prédire", type="primary", use_container_width=True)

    if predict_clicked:
        if not endpoint_url or not endpoint_key:
            st.error("Configuration manquante: renseigne l'URL et la clé de l'endpoint dans la barre latérale.")
        else:
            try:
                with st.spinner("Appel du endpoint Azure ML..."):
                    result = call_endpoint([sepal_length, sepal_width, petal_length, petal_width])

                prediction_class = CLASS_NAMES[result["prediction"]]
                probabilities = result["probabilities"]

                st.success(f"Classe prédite: **{prediction_class}**")
                st.bar_chart({"probabilité": dict(zip(CLASS_NAMES, probabilities))})

            except requests.exceptions.Timeout:
                st.error("Le endpoint n'a pas répondu à temps (timeout).")
            except requests.exceptions.HTTPError as e:
                st.error(f"Erreur HTTP {e.response.status_code}: {e.response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Impossible de contacter le endpoint: {e}")
            except (KeyError, IndexError, ValueError) as e:
                st.error(f"Réponse inattendue du endpoint: {e}")

with tab_batch:
    st.subheader("Import CSV ou Excel")
    st.caption(
        "Le fichier doit contenir 4 colonnes reconnaissables (longueur/largeur de "
        "sépale et de pétale), ex: `sepal_length, sepal_width, petal_length, petal_width`."
    )

    uploaded_file = st.file_uploader("Choisir un fichier", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Impossible de lire le fichier: {e}")
            input_df = None

        if input_df is not None and input_df.empty:
            st.warning("Le fichier ne contient aucune ligne.")
        elif input_df is not None:
            try:
                column_map = map_columns(input_df.columns)
            except ValueError as e:
                st.error(str(e))
                column_map = None

            if column_map:
                st.write(f"{len(input_df)} ligne(s) détectée(s).")
                st.dataframe(input_df.head(), use_container_width=True)
                if len(input_df) > 200:
                    st.info("Fichier volumineux: une requête est envoyée par ligne, la prédiction peut prendre du temps.")

                predict_batch_clicked = st.button(
                    "🔮 Prédire tout le fichier", type="primary", use_container_width=True
                )

                if predict_batch_clicked:
                    if not endpoint_url or not endpoint_key:
                        st.error("Configuration manquante: renseigne l'URL et la clé de l'endpoint dans la barre latérale.")
                    else:
                        rows = []
                        error_count = 0
                        progress = st.progress(0.0, text="Prédiction en cours...")

                        for position, (_, row) in enumerate(input_df.iterrows(), start=1):
                            record = {"prediction": None, "confiance": None}
                            record.update({name: None for name in CLASS_NAMES})
                            try:
                                features = [float(row[column_map[f]]) for f in FEATURE_ORDER]
                                result = call_endpoint(features)
                                probabilities = result["probabilities"]
                                record["prediction"] = CLASS_NAMES[result["prediction"]]
                                record["confiance"] = max(probabilities)
                                record.update(dict(zip(CLASS_NAMES, probabilities)))
                            except Exception:
                                error_count += 1
                            rows.append(record)
                            progress.progress(position / len(input_df))

                        progress.empty()

                        results_df = pd.concat(
                            [input_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1
                        )

                        success_count = len(input_df) - error_count
                        if error_count:
                            st.warning(f"{error_count} ligne(s) n'ont pas pu être prédite(s) (colonnes 'prediction' vides).")
                        st.success(f"{success_count}/{len(input_df)} prédictions réussies.")

                        display_df = results_df.copy()
                        for col in ["confiance"] + CLASS_NAMES:
                            if col in display_df.columns:
                                display_df[col] = display_df[col].map(
                                    lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "—"
                                )
                        st.dataframe(display_df, use_container_width=True)

                        if success_count:
                            st.bar_chart(results_df["prediction"].value_counts())

                        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Télécharger les résultats (CSV)",
                            data=csv_bytes,
                            file_name="predictions_iris.csv",
                            mime="text/csv",
                        )
