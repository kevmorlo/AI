from apiflask import APIFlask, Schema, abort
from apiflask.fields import Integer, Float, Dict, List, Nested
from apiflask.validators import Range
import os
import pandas as pd
import numpy as np
from model_accident import convert_grav, convert_annee, lire_fichier_csv
from sklearn.linear_model import LinearRegression
import joblib
from flask import request, redirect

# Remplacer Flask par APIFlask
app = APIFlask(__name__, 
               title="API des Accidents",
               version="1.0.0", 
               spec_path='/api/swagger',
               docs_path='/api/docs')

# Définir les schémas pour la documentation
class AccidentSchema(Schema):
    id = Integer(required=True, description="Identifiant unique de l'accident")
    annee = Integer(required=True, description="Année de l'accident")
    gravite = Integer(required=True, description="Niveau de gravité (1-100)")

class AccidentCreateSchema(Schema):
    annee = Integer(required=True, description="Année de l'accident", validate=Range(min=1900, max=2100))
    gravite = Integer(required=True, description="Niveau de gravité", validate=Range(min=1, max=100))

class AccidentListSchema(Schema):
    accidents = List(Nested(AccidentSchema), required=True, description="Liste des accidents")
    total = Integer(required=True, description="Nombre total d'accidents")
    pages = Integer(required=True, description="Nombre total de pages")
    current_page = Integer(required=True, description="Page actuelle")

class PredictionSchema(Schema):
    annee = Integer(required=True, description="Année de prédiction")
    gravite_predite = Float(required=True, description="Gravité prédite")

class StatsSchema(Schema):
    total_accidents = Integer(required=True, description="Nombre total d'accidents")
    annee_moyenne = Float(required=True, description="Année moyenne des accidents")
    gravite_moyenne = Float(required=True, description="Gravité moyenne des accidents")

# Structure de données en mémoire pour stocker les accidents (pas de changement)
class AccidentManager:
    def __init__(self):
        self.accidents = []
        self.id_counter = 1
        self.model = None
        
    def load_data_from_csv(self):
        """Charge les données depuis les fichiers CSV"""
        print("Chargement des données depuis les fichiers CSV...")
        
        # Charger les données des usagers depuis le CSV
        data_usagers = lire_fichier_csv("data/usagers-2023.csv")
        
        # Suppression des lignes trop courtes
        data_usagers = [d for d in data_usagers if len(d) > 8]
        
        # Conversion et chargement des données
        for d in data_usagers:
            try:
                annee = convert_annee(d[8][1:-1])
                gravite = convert_grav(d[6][1:-1])
                
                # Ne garder que les données cohérentes
                if annee > -1 and gravite > -1:
                    self.add_accident(annee, gravite)
            except (IndexError, ValueError):
                # Ignorer les lignes problématiques
                continue
                
        print(f"Données chargées : {len(self.accidents)} accidents")
        
        # Entraîner le modèle de prédiction
        self.train_prediction_model()
        
    def train_prediction_model(self):
        """Entraîne un modèle de régression linéaire pour prédire la gravité en fonction de l'année"""
        if len(self.accidents) == 0:
            print("Pas de données pour entraîner le modèle")
            return
            
        print("Entraînement du modèle de prédiction...")
        
        # Préparation des données
        x_annee = np.array([[a['annee']] for a in self.accidents])
        y_gravite = np.array([a['gravite'] for a in self.accidents])
        
        # Entraînement du modèle
        self.model = LinearRegression()
        self.model.fit(x_annee, y_gravite)
        
        print("Modèle entraîné avec succès")
        
    def predict_gravite(self, annee):
        """Prédit la gravité pour une année donnée"""
        if self.model is None:
            return {"error": "Le modèle n'a pas été entraîné"}
            
        prediction = self.model.predict([[annee]])[0]
        return {
            "annee": annee,
            "gravite_predite": round(prediction, 2)
        }
        
    def add_accident(self, annee, gravite):
        accident = {
            'id': self.id_counter,
            'annee': annee,
            'gravite': gravite
        }
        self.accidents.append(accident)
        self.id_counter += 1
        return accident
        
    def get_all_accidents(self, page=1, per_page=100):
        start = (page - 1) * per_page
        end = start + per_page
        items = self.accidents[start:end]
        return {
            'items': items,
            'total': len(self.accidents),
            'pages': (len(self.accidents) // per_page) + (1 if len(self.accidents) % per_page else 0),
            'page': page
        }
        
    def get_accident_by_id(self, id):
        for accident in self.accidents:
            if accident['id'] == id:
                return accident
        return None
        
    def get_stats(self):
        if not self.accidents:
            return {'total_accidents': 0, 'annee_moyenne': 0, 'gravite_moyenne': 0}
            
        annee_sum = sum(a['annee'] for a in self.accidents)
        gravite_sum = sum(a['gravite'] for a in self.accidents)
        count = len(self.accidents)
        
        return {
            'total_accidents': count,
            'annee_moyenne': round(annee_sum / count, 2) if count else 0,
            'gravite_moyenne': round(gravite_sum / count, 2) if count else 0
        }

# Initialisation du gestionnaire d'accidents
accident_manager = AccidentManager()

# Route par défaut redirigeant vers la documentation Swagger
@app.get('/')
@app.doc(summary='Page d\'accueil', description='Redirige vers la documentation Swagger')
def redirict_to_docs():
    return redirect('/api/docs')

# Routes API avec documentation Swagger
@app.get('/api/accidents')
@app.doc(summary='Récupérer tous les accidents', description='Retourne une liste paginée des accidents')
@app.output(AccidentListSchema)
def get_accidents():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 100, type=int)
    
    accidents = accident_manager.get_all_accidents(page=page, per_page=per_page)
    
    return {
        'accidents': accidents['items'],
        'total': accidents['total'],
        'pages': accidents['pages'],
        'current_page': accidents['page']
    }

@app.get('/api/accidents/<int:id>')
@app.doc(summary='Récupérer un accident par ID', description='Retourne les détails d\'un accident spécifique')
@app.output(AccidentSchema)
def get_accident(id):
    accident = accident_manager.get_accident_by_id(id)
    if not accident:
        abort(404, 'Accident non trouvé')
    return accident

@app.post('/api/accidents')
@app.doc(summary='Créer un nouvel accident', description='Ajoute un nouvel accident à la base de données')
@app.input(AccidentCreateSchema)
@app.output(AccidentSchema, status_code=201)
def create_accident(data):
    accident = accident_manager.add_accident(data['annee'], data['gravite'])
    return accident

@app.get('/api/predict/<int:annee>')
@app.doc(summary='Prédire la gravité pour une année', 
         description='Utilise un modèle de régression linéaire pour prédire la gravité des accidents pour une année donnée')
@app.output(PredictionSchema)
def predict_gravite(annee):
    result = accident_manager.predict_gravite(annee)
    return result

@app.get('/api/stats')
@app.doc(summary='Obtenir des statistiques', description='Retourne des statistiques générales sur les accidents')
@app.output(StatsSchema)
def get_stats():
    stats = accident_manager.get_stats()
    return stats

if __name__ == '__main__':
    # Charger les données avant de démarrer l'application
    accident_manager.load_data_from_csv()
    app.run(debug=True)