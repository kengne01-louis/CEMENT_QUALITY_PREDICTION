from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Autorise les requêtes cross-origin

# Charger votre modèle Random Forest pour régression
try:
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("✅ Modèle de régression chargé avec succès")
    
    # Afficher les informations du modèle
    if hasattr(model, 'n_features_in_'):
        print(f"📊 Nombre de features attendues: {model.n_features_in_}")
    if hasattr(model, 'n_estimators'):
        print(f"🌳 Nombre d'arbres: {model.n_estimators}")
        
except FileNotFoundError:
    print("❌ Fichier modèle non trouvé")
    model = None
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    model = None

# Noms des features dans l'ordre exact
FEATURE_NAMES = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']

# Route de test
@app.route('/')
def home():
    return jsonify({
        "message": "API Random Forest Regression pour la Résistance du Béton",
        "problem_type": "REGRESSION",
        "expected_features": 8,
        "feature_names": FEATURE_NAMES,
        "units": {
            "cement": "kg/m³",
            "slag": "kg/m³", 
            "ash": "kg/m³",
            "water": "kg/m³",
            "superplastic": "kg/m³",
            "coarseagg": "kg/m³",
            "fineagg": "kg/m³",
            "age": "jours",
            "prediction": "MPa"
        },
        "endpoints": {
            "test": "GET /",
            "prediction": "POST /predict",
            "batch_prediction": "POST /predict_batch", 
            "santé": "GET /health",
            "model_info": "GET /model_info",
            "validate_features": "POST /validate_features"
        }
    })

# Route de santé
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_type": "RandomForestRegressor" if model else None,
        "feature_count": len(FEATURE_NAMES) if model else 0
    })

# Route pour les informations du modèle
@app.route('/model_info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({"error": "Modèle non chargé"}), 500
    
    info = {
        "model_type": type(model).__name__,
        "n_estimators": getattr(model, 'n_estimators', 'N/A'),
        "n_features": getattr(model, 'n_features_in_', 'N/A'),
        "max_depth": getattr(model, 'max_depth', 'N/A'),
        "problem_type": "REGRESSION",
        "feature_names": FEATURE_NAMES
    }
    
    # Importance des features 
    if hasattr(model, 'feature_importances_'):
        feature_importance_dict = {}
        for i, (name, importance) in enumerate(zip(FEATURE_NAMES, model.feature_importances_)):
            feature_importance_dict[name] = float(importance)
        
        info["feature_importance"] = feature_importance_dict
    
    return jsonify(info)

# Route de prédiction simple 
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Vérifier si le modèle est chargé
        if model is None:
            return jsonify({"error": "Modèle non chargé"}), 500
        
        # Récupérer les données JSON
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Aucune donnée fournie"}), 400
        
        # Vérifier la présence des features
        if 'features' not in data:
            return jsonify({
                "error": "Clé 'features' manquante",
                "format_requis": {
                    "features": [cement, slag, ash, water, superplastic, coarseagg, fineagg, age]
                },
                "feature_names": FEATURE_NAMES
            }), 400
        
        features = data['features']
        
        # Valider le nombre de features
        if len(features) != 8:
            return jsonify({
                "error": f"Nombre de features incorrect. Attendu: 8, Reçu: {len(features)}",
                "feature_names": FEATURE_NAMES,
                "features_received": features
            }), 400
        
        # Convertir en array numpy et reshape pour la prédiction
        features_array = np.array(features).reshape(1, -1)
        
        # Faire la prédiction
        prediction = model.predict(features_array)
        
        # Pour la régression, on peut aussi obtenir des intervalles de confiance
        # en utilisant les arbres individuels
        if hasattr(model, 'estimators_'):
            tree_predictions = []
            for tree in model.estimators_:
                tree_pred = tree.predict(features_array)
                tree_predictions.append(tree_pred[0])
            
            confidence_interval = {
                "mean": float(np.mean(tree_predictions)),
                "std": float(np.std(tree_predictions)),
                "min": float(np.min(tree_predictions)),
                "max": float(np.max(tree_predictions)),
                "confidence_95_lower": float(np.percentile(tree_predictions, 2.5)),
                "confidence_95_upper": float(np.percentile(tree_predictions, 97.5))
            }
        else:
            confidence_interval = None
        
        # Préparer la réponse avec les noms des features
        features_dict = {name: value for name, value in zip(FEATURE_NAMES, features)}
        
        response = {
            "prediction": float(prediction[0]),
            "prediction_unit": "MPa",
            "features_used": features_dict,
            "model_type": "regression"
        }
        
        # Ajout de l'intervalle de confiance 
        if confidence_interval:
            response["confidence_interval"] = confidence_interval
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction: {str(e)}"}), 500

# Route pour les prédictions multiple 
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if model is None:
            return jsonify({"error": "Modèle non chargé"}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Aucune donnée fournie"}), 400
        
        if 'samples' not in data:
            return jsonify({
                "error": "Clé 'samples' manquante",
                "format_requis": {
                    "samples": [
                        [cement, slag, ash, water, superplastic, coarseagg, fineagg, age],
                        [cement, slag, ash, water, superplastic, coarseagg, fineagg, age]
                    ]
                },
                "feature_names": FEATURE_NAMES
            }), 400
        
        samples = data['samples']
        
        # Valider chaque échantillon
        for i, sample in enumerate(samples):
            if len(sample) != 8:
                return jsonify({
                    "error": f"Échantillon {i} a {len(sample)} features. Attendu: 8",
                    "feature_names": FEATURE_NAMES,
                    "sample_index": i,
                    "sample_received": sample
                }), 400
        
        # Convertir en array numpy
        features_array = np.array(samples)
        
        # Faire les prédictions
        predictions = model.predict(features_array)
        
        # Préparer la réponse détaillée
        results = []
        for i, (sample, pred) in enumerate(zip(samples, predictions)):
            features_dict = {name: value for name, value in zip(FEATURE_NAMES, sample)}
            results.append({
                "sample_index": i,
                "prediction": float(pred),
                "prediction_unit": "MPa",
                "features": features_dict
            })
        
        response = {
            "predictions": [float(pred) for pred in predictions],
            "prediction_unit": "MPa",
            "results": results,
            "count": len(predictions),
            "statistics": {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions))
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Erreur lors des prédictions batch: {str(e)}"}), 500

# Route pour valider les features
@app.route('/validate_features', methods=['POST'])
def validate_features():
    data = request.get_json()
    
    if not data or 'features' not in data:
        return jsonify({"valid": False, "error": "Données ou clé 'features' manquante"})
    
    features = data['features']
    
    if len(features) != 8:
        return jsonify({
            "valid": False, 
            "error": f"Nombre de features incorrect. Attendu: 8, Reçu: {len(features)}",
            "feature_names": FEATURE_NAMES
        })
    
    # Vérifier que toutes les features sont numériques
    try:
        [float(f) for f in features]  # Test de conversion
        
        # Créer un dictionnaire avec les noms des features
        features_dict = {name: value for name, value in zip(FEATURE_NAMES, features)}
        
        return jsonify({
            "valid": True,
            "message": "Features valides",
            "features_count": len(features),
            "features": features_dict
        })
    except ValueError:
        return jsonify({
            "valid": False,
            "error": "Toutes les features doivent être numériques"
        })

# Route pour obtenir la documentation complète
@app.route('/docs', methods=['GET'])
def documentation():
    return jsonify({
        "api_documentation": {
            "description": "API pour la prédiction de la résistance du béton using Random Forest",
            "features": FEATURE_NAMES,
            "endpoints": {
                "GET /": "Page d'accueil avec informations générales",
                "GET /health": "Statut de santé de l'API",
                "GET /model_info": "Informations détaillées du modèle",
                "GET /docs": "Cette documentation",
                "POST /predict": {
                    "description": "Prédiction simple",
                    "body": {
                        "features": "Liste de 8 valeurs numériques dans l'ordre: cement, slag, ash, water, superplastic, coarseagg, fineagg, age"
                    },
                    "response": {
                        "prediction": "Valeur prédite en MPa",
                        "features_used": "Dictionnaire des features utilisées",
                        "confidence_interval": "Intervalle de confiance (si disponible)"
                    }
                },
                "POST /predict_batch": {
                    "description": "Prédictions multiples",
                    "body": {
                        "samples": "Liste de listes, chaque sous-liste contient 8 valeurs dans le même ordre"
                    },
                    "response": {
                        "predictions": "Liste des prédictions en MPa",
                        "results": "Détails par échantillon",
                        "statistics": "Statistiques des prédictions"
                    }
                },
                "POST /validate_features": {
                    "description": "Validation des features",
                    "body": {
                        "features": "Liste de 8 valeurs à valider"
                    },
                    "response": {
                        "valid": "Booléen indiquant si les features sont valides",
                        "features": "Dictionnaire des features avec leurs noms"
                    }
                }
            }
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)