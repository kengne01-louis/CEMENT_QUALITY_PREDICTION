import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
import gdown


# Couleurs des pages
def set_background(page):
    if page == "Accueil":
        bg_color = "#5A37374D"  
        accent = "#15C08485"
    elif page == "Prédiction":
        bg_color = "#E8F5EA68"  
        accent = "#352E7DF1"
    elif page == "Appel API":
        bg_color = "#E8F4FD"  
        accent = "#1E88E5"
    elif page == "Visualisations":
        bg_color = "#FFE9E0FF" 
        accent = "#EF6C00"
    elif page == "À propos":
        bg_color = "#F3E5F5"  
        accent = "#8E24AA"
    else:
        bg_color = "#F0F4C3"  
        accent = "#827717"
    
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {bg_color};
            color: #1A237E;
        }}
        /* Titres */
        h1, h2, h3, h4, h5, h6 {{
            color: {accent};
        }}
        /* Champs de saisie */
        .stNumberInput input {{
            background-color: #FFFFFFDD;
            color: #1B1B1B;
            font-weight: 600;
            border-radius: 10px;
            border: 2px solid {accent};
        }}
        /* Boutons */
        .stButton>button {{
            background-color: {accent};
            color: white;
            border-radius: 10px;
            padding: 8px 18px;
            font-weight: bold;
            font-size: 16px;
        }}
        .stButton>button:hover {{
            background-color: #0D47A1;
            color: #fff;
        }}
        /* Tableaux et données */
        div[data-testid="stDataFrame"] {{
            background-color: #FFFFFFCC;
            border-radius: 10px;
            padding: 10px;
        }}
        /* Graphiques */
        .js-plotly-plot .plotly {{
            background-color: transparent !important;
        }}
        </style>
    """, unsafe_allow_html=True)

sidebar_css = """
    <style>
    [data-testid="stSidebar"] {
        background-color: #1E3A8A;
    }
    
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio div,
    [data-testid="stSidebar"] .stTitle,
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        color: white;
    }
    </style>
    """
st.markdown(sidebar_css, unsafe_allow_html=True)

# Fonctions pour l'API
def test_api_connection():
    """Tester la connexion à l'API"""
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            return True, "✅ API connectée avec succès"
        else:
            return False, f"❌ API retourne une erreur: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "❌ Impossible de se connecter à l'API. Vérifiez qu'elle est lancée sur le port 5000."
    except Exception as e:
        return False, f"❌ Erreur de connexion: {str(e)}"

def predict_via_api(features):
    """Faire une prédiction via l'API"""
    try:
        response = requests.post(
            'http://localhost:5000/predict',
            json={'features': features},
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Erreur API ({response.status_code}): {response.json().get('error', 'Unknown error')}"}
    
    except requests.exceptions.ConnectionError:
        return {"error": "❌ Impossible de se connecter à l'API"}
    except requests.exceptions.Timeout:
        return {"error": "⏰ Timeout - L'API met trop de temps à répondre"}
    except Exception as e:
        return {"error": f"🚨 Erreur: {str(e)}"}

def predict_batch_via_api(samples):
    """Faire des prédictions multiples via l'API"""
    try:
        response = requests.post(
            'http://localhost:5000/predict_batch',
            json={'samples': samples},
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Erreur API ({response.status_code}): {response.json().get('error', 'Unknown error')}"}
    
    except Exception as e:
        return {"error": f"Erreur lors des prédictions batch: {str(e)}"}

def get_model_info_api():
    """Récupérer les informations du modèle depuis l'API"""
    try:
        response = requests.get('http://localhost:5000/model_info', timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"error": "Impossible de récupérer les infos du modèle"}
    except:
        return {"error": "API non disponible"}

# Titre de l'application
st.title("APPLICATION DU MACHINE LEARNING POUR L'ESTIMATION DE LA RESISTANCE DU BETON.")
st.markdown("""
LE BUT PRINCIPAL EST DE PREDIRE LA RESISTANCE DU BETON.
""")

# Sidebar avec logo
with st.sidebar:
    # Ajout du logo
    try:
        logo = Image.open("deco.jpg")
        st.image(logo, width=300)
    except Exception as e:
        st.error(f"Logo non trouvé: {e}")

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Accueil", "Prédiction", "Appel API", "Visualisations", "À propos"])

set_background(page)



# Fonction pour charger le modèle
def load_model():
    try:
        model = joblib.load('random_forest_model.pkl','rb')
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None 



# Page d'accueil
if page == "Accueil":
    st.header("🏠 Accueil")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Description et Modele")
        st.markdown("""
        La résistance du béton est un facteur essentiel pour garantir la durabilité et la sécurité des constructions.  
        Elle dépend de plusieurs composants. Dans notre contexte, nous avons texté deux modèles:
        DECISION TREE ET RANDOM FOREST dont **Random Forest** est notre meilleur modele.

        Les variables utilisées sont :

        - **cement** : Quantité de ciment (en kg/m³)  
        - **slag** : Quantité de laitier (en kg/m³)  
        - **ash** : Quantité de cendres volantes (en kg/m³)  
        - **water** : Quantité d'eau (en kg/m³)  
        - **superplastic** : Quantité de superplastifiant (en kg/m³)  
        - **coarseagg** : Quantité d'agrégats grossiers (en kg/m³)  
        - **fineagg** : Quantité d'agrégats fins (en kg/m³)  
        - **age** : Âge du béton (en jours)  
        - **strength** : Résistance à la compression (en MPa)
                    
        Le modele utilisé ici est RANDOM FOREST, pour prédire la résistance du béton en fonction des autres variables.
    """)

    with col2:
        st.subheader("Suivez les instructions")
        st.markdown("""
        1. Allez dans l'onglet **Prédiction** pour utiliser le modèle local
        2. Allez dans l'onglet **Appel API** pour tester via l'API REST
        3. Entrez les valeurs des features
        4. Cliquez sur **Prédire**
        5. Visualisez les résultats
        """)
    
    # Afficher les informations du modèle chargé
    model = load_model()
    if model is not None:
        st.success("✅ Modèle local chargé avec succès!")
        st.info(f"Nombre d'arbres dans la forêt: {model.n_estimators}")

# Page de prédiction locale
elif page == "Prédiction":
    st.header("✍👇✍ Prédiction Locale")
    
    model = load_model()
    
    if model is not None:
        st.subheader("Entrez les valeurs des features")
        
        input_method = st.radio("Choisissez la méthode de saisie:", 
                                  ["Formulaire", "À partir d'un fichier CSV"]) 
        
        if input_method == "Formulaire":
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cement = st.number_input("cement (kg/m³)", value=0.0, min_value=0.0)
                slag = st.number_input("slag (kg/m³)", value=0.0, min_value=0.0)
                
            with col2:
                ash = st.number_input("ash (kg/m³)", value=0.0, min_value=0.0)
                water = st.number_input("water (kg/m³)", value=0.0, min_value=0.0)
                
            with col3:
                superplastic = st.number_input("superplastic (kg/m³)", value=0.0, min_value=0.0)
                coarseagg = st.number_input("coarseagg (kg/m³)", value=0.0, min_value=0.0)
            
            with col4:
                fineagg = st.number_input("fineagg (kg/m³)", value=0.0, min_value=0.0)
                age = st.number_input("age (jours)", value=1, min_value=1)
            
            input_data = np.array([[cement, slag, ash, water, superplastic, coarseagg, fineagg, age]])
            
            if st.button("Faire la Prédiction", type="primary"):
                try:
                    prediction = model.predict(input_data)
                    st.success(f"📗👀 **Prédiction:** {prediction[0]:.4f} MPa")
                    
                    with st.expander("Détails de la prédiction"):
                        st.write(f"**Valeurs d'entrée:**")
                        st.write(f"- Cement: {cement} kg/m³")
                        st.write(f"- Slag: {slag} kg/m³")
                        st.write(f"- Ash: {ash} kg/m³")
                        st.write(f"- Water: {water} kg/m³")
                        st.write(f"- Superplastic: {superplastic} kg/m³")
                        st.write(f"- Coarseagg: {coarseagg} kg/m³")
                        st.write(f"- Fineagg: {fineagg} kg/m³")
                        st.write(f"- Age: {age} jours")
                        st.write(f"**Modèle utilisé:** Random Forest ({model.n_estimators} arbres)")
                        
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction: {e}")
        
        else:
            st.subheader("Importer le fichier CSV")
            uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("Aperçu des données chargées:")
                    st.dataframe(df.head())
                    
                    if st.button("Prédire sur le fichier", type="primary"):
                        predictions = model.predict(df)
                        df['Prediction_Resistance_MPa'] = predictions
                        
                        st.success("Prédictions terminées!")
                        st.write("Résultats:")
                        st.dataframe(df)
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📙✍ Télécharger les prédictions",
                            data=csv,
                            file_name="predictions_local.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Erreur lors du traitement du fichier: {e}")

# Page Appel API
elif page == "Appel API":
    st.header("🌐 Prédiction via API REST")
    
    # Test de connexion à l'API
    with st.expander("🔻👇 Test de Connexion à l'API 👇🔻"):
        if st.button("Tester la connexion à l'API"):
            status, message = test_api_connection()
            if status:
                st.success(message)
                
                # Afficher les infos du modèle API
                model_info = get_model_info_api()
                if "error" not in model_info:
                    st.info(f"**Modèle API:** {model_info.get('model_type', 'N/A')}")
                    st.info(f"**Arbres:** {model_info.get('n_estimators', 'N/A')}")
                    st.info(f"**Features:** {model_info.get('n_features', 'N/A')}")
            else:
                st.error(message)
    
    st.subheader(" Prédiction Simple via API")
    
    # Formulaire pour la prédiction simple
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cement = st.number_input("cement (kg/m³)", value=0.0, min_value=0.0, key="api_cement")
        slag = st.number_input("slag (kg/m³)", value=0.0, min_value=0.0, key="api_slag")
                
    with col2:
        ash = st.number_input("ash (kg/m³)", value=0.0, min_value=0.0, key="api_ash")
        water = st.number_input("water (kg/m³)", value=0.0, min_value=0.0, key="api_water")
                
    with col3:
        superplastic = st.number_input("superplastic (kg/m³)", value=0.0, min_value=0.0, key="api_superplastic")
        coarseagg = st.number_input("coarseagg (kg/m³)", value=0.0, min_value=0.0, key="api_coarseagg")
    
    with col4:
        fineagg = st.number_input("fineagg (kg/m³)", value=0.0, min_value=0.0, key="api_fineagg")
        age = st.number_input("age (jours)", value=1, min_value=1, key="api_age")
    
    features = [cement, slag, ash, water, superplastic, coarseagg, fineagg, age]
    
    if st.button(" Prédire via API", type="primary"):
        with st.spinner("Prédiction en cours via l'API..."):
            result = predict_via_api(features)
        
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(f"✍📙 **Prédiction via API:** {result['prediction']:.4f} MPa")
            
            # Affichage détaillé
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✍📙 Résistance Prédite", f"{result['prediction']:.4f} MPa")
            
            # Intervalle de confiance si disponible
            if "confidence_interval" in result:
                conf = result["confidence_interval"]
                with col2:
                    st.metric("⏹ Écart-type", f"{conf['std']:.4f}")
                with col3:
                    st.metric("📐 Intervalle 95%", 
                             f"[{conf['confidence_95_lower']:.2f}, {conf['confidence_95_upper']:.2f}]")
            
            # Détails complets
            with st.expander("📗 Détails de la réponse API 📗 "):
                st.json(result)
    
    st.subheader("🔻 Prédictions Multiple via API 🔻")
    
    st.markdown("""
    Téléchargez un fichier CSV avec les 8 colonnes dans cet ordre:
    `cement,slag,ash,water,superplastic,coarseagg,fineagg,age`
    """)
    
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv", key="api_file")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("**Aperçu des données chargées:**")
            st.dataframe(df.head())
            
            # Vérifier les colonnes
            expected_columns = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']
            if all(col in df.columns for col in expected_columns):
                st.success("✅ Format CSV valide")
                
                if st.button(" 📙 Prédire le Batch via API", type="primary"):
                    samples = df[expected_columns].values.tolist()
                    
                    with st.spinner(f"Prédiction de {len(samples)} échantillons via l'API..."):
                        results = predict_batch_via_api(samples)
                    
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        # Ajouter les prédictions au DataFrame
                        df['Prediction_Resistance_MPa'] = results['predictions']
                        
                        st.success(f"✅ {len(samples)} prédictions via API réussies !")
                        
                        # Afficher les résultats
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Résultats détaillés:**")
                            st.dataframe(df)
                        
                        with col2:
                            st.write("**Statistiques:**")
                            stats = results.get('statistics', {})
                            st.metric("Moyenne", f"{stats.get('mean', 0):.4f} MPa")
                            st.metric("Écart-type", f"{stats.get('std', 0):.4f} MPa")
                            st.metric("Minimum", f"{stats.get('min', 0):.4f} MPa")
                            st.metric("Maximum", f"{stats.get('max', 0):.4f} MPa")
                        
                        # Téléchargement des résultats
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Télécharger les résultats CSV",
                            data=csv,
                            file_name="predictions_api.csv",
                            mime="text/csv"
                        )
            else:
                st.error("❌ Le fichier CSV doit contenir les colonnes: cement,slag,ash,water,superplastic,coarseagg,fineagg,age")
                
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier: {str(e)}")
    
    # Documentation de l'API
    with st.expander("📚 Documentation de l'API 📚"):
        st.markdown("""
        **Endpoints disponibles:**
        - `GET /` - Page d'accueil
        - `GET /health` - Santé de l'API
        - `GET /model_info` - Informations du modèle
        - `POST /predict` - Prédiction simple
        - `POST /predict_batch` - Prédictions multiples
        
        **Format JSON pour la prédiction simple:**
        ```json
        {
            "features": [cement, slag, ash, water, superplastic, coarseagg, fineagg, age]
        }
        ```
        
        **Format JSON pour les prédictions multiples:**
        ```json
        {
            "samples": [
                [c1, s1, a1, w1, sp1, ca1, fa1, age1],
                [c2, s2, a2, w2, sp2, ca2, fa2, age2]
            ]
        }
        ```
        """)

# Page de visualisations
elif page == "Visualisations":
    st.header("📊🔻 Visualisations 🔻📊")
    
    model = load_model()
    
    if model is not None:
        st.subheader("Importance des Features")
        
        if hasattr(model, 'feature_importances_'):
            feature_names = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=feature_importance, x='importance', y='feature', ax=ax)
            ax.set_title('Importance des Features pour la Résistance du Béton')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Features')
            st.pyplot(fig)
            
            st.write("Détail de l'importance des features:")
            st.dataframe(feature_importance)
        else:
            st.warning("Impossible d'afficher l'importance des features pour ce modèle.")
        
        st.subheader("Informations du Modèle")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Nombre d'arbres", model.n_estimators)
            st.metric("Profondeur max", str(model.max_depth) if model.max_depth else "None")
            
        with col2:
            st.metric("Samples split min", model.min_samples_split)
            st.metric("Samples leaf min", model.min_samples_leaf)

# Page À propos
elif page == "À propos":
    st.header("ℹ️ À propos")
    
    st.markdown("""
    ### Application de Déploiement Random Forest
    
    **Fonctionnalités:**
    - 📊 Prédictions en temps réel (Local et API)
    - 🌐 API REST pour intégration
    - 📈 Visualisation de l'importance des features
    - 📁 Support des fichiers CSV
    - 🎯 Interface utilisateur intuitive
    
    **Quelques technologies:**
    - Streamlit pour l'interface
    - Flask pour l'API REST
    - Scikit-learn pour le machine learning
    - Random Forest pour la régression
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.markdown("Téléphone: 659 060 681")
st.sidebar.markdown("Email: louiskngn@gmail.com")
st.sidebar.markdown("---")
