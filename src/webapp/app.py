"""
Interface Streamlit pour consommer le endpoint Azure ML Managed Online
Endpoint qui héberge le modèle Iris (déployé via scripts/deploy_endpoint.py
et terraform/main.tf).

Lancement local:
    streamlit run src/webapp/app.py

Configuration (variables d'environnement ou fichier .env à la racine):
    AZURE_ML_ENDPOINT_URL   URL de scoring de l'endpoint (scoring_uri)
    AZURE_ML_ENDPOINT_KEY   Clé d'authentification de l'endpoint
"""

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ordre des classes retourné par score.py: model.predict() renvoie l'index
# tel qu'encodé par sklearn.datasets.load_iris (0=setosa, 1=versicolor,
# 2=virginica) — voir src/train/train.py.
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

SAMPLES = {
    "Setosa (exemple)": (5.1, 3.5, 1.4, 0.2),
    "Versicolor (exemple)": (6.0, 2.7, 4.2, 1.3),
    "Virginica (exemple)": (6.9, 3.1, 5.4, 2.1),
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
        payload = {"data": [[sepal_length, sepal_width, petal_length, petal_width]]}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {endpoint_key}",
        }
        try:
            with st.spinner("Appel du endpoint Azure ML..."):
                response = requests.post(endpoint_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                st.error(f"Le endpoint a renvoyé une erreur: {result['error']}")
            else:
                prediction_idx = result["prediction"]
                probabilities = result["probabilities"]
                prediction_class = CLASS_NAMES[prediction_idx]

                st.success(f"Classe prédite: **{prediction_class}**")
                st.bar_chart({"probabilité": dict(zip(CLASS_NAMES, probabilities))})

        except requests.exceptions.Timeout:
            st.error("Le endpoint n'a pas répondu à temps (timeout).")
        except requests.exceptions.HTTPError as e:
            st.error(f"Erreur HTTP {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Impossible de contacter le endpoint: {e}")
        except (KeyError, IndexError, ValueError) as e:
            st.error(f"Réponse inattendue du endpoint: {e}")
