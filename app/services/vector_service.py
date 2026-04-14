import json
import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Définition du chemin absolu vers nos fausses données générées précédemment
# Ajusté pour pointer vers les artifacts du workspace
# Définition du chemin vers les données de test locales
MOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "plaintes_fictives.json")

class VectorSearchService:
    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialise le modèle NLP pre-entrainé multilingue (efficace en français) 
        et l'index FAISS.
        """
        print(f"Chargement du modèle {model_name}... (cela peut prendre quelques secondes)")
        self.model = SentenceTransformer(model_name)
        
        # L2 distance index (le plus standard pour les embeddings)
        # La dimension dépend du modèle choisi (384 pour MiniLM)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        
        self.metadata = pd.DataFrame() # Stocke les infos (solutions) liées aux vecteurs
        self._load_and_index_mock_data()

    def _load_and_index_mock_data(self):
        """Charge les données JSON fictives et calcule leurs vecteurs."""
        try:
            with open(MOCK_DATA_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                print("Aucune donnée trouvée dans le JSON.")
                return
                
            self.metadata = pd.DataFrame(data)
            
            # On veut que l'IA compare les descriptions de plaintes
            texts_to_embed = self.metadata['texte_plainte'].tolist()
            
            print(f"Indexation de {len(texts_to_embed)} éléments de l'historique...")
            # Encodage de tous les textes en vecteurs mathématiques (matrices)
            embeddings = self.model.encode(texts_to_embed, convert_to_numpy=True)
            
            # Ajout des vecteurs dans notre base de données "en mémoire" FAISS
            self.index.add(embeddings)
            print("Base de données vectorielle initialisée avec succès.")
            
        except Exception as e:
            print(f"Erreur lors de l'initialisation de FAISS: {e}")

    def search_similar(self, query: str, top_k: int = 3, category_filter: str = None) -> list[dict]:
        """
        Recherche les k plaintes les plus sémantiquement proches de la requête.
        Dans une V2, on pourra pré-filtrer par catégorie avec FAISS.
        """
        if self.index.ntotal == 0:
            return []

        # 1. on transforme la plainte de l'agent en vecteur
        query_vector = self.model.encode([query], convert_to_numpy=True)
        
        # 2. Chercher dans FAISS les plus proches
        # La distance plus petite = plus proche sémantiquement
        distances, indices = self.index.search(query_vector, k=top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: # Si pas assez de résultats trouvés
                continue
                
            # Retrouver les métadonnées (statut, solution, etc.) de la plainte trouvée
            row = self.metadata.iloc[idx].to_dict()
            
            # Simple filtrage par catégorie (post-recherche pour ce MVP simple)
            if category_filter and str(row.get('objet_categorie', '')).lower() != category_filter.lower():
                continue
                
            results.append({
                "score_similarite": float(distances[0][i]),
                "id_historique": int(row['id']),
                "categorie": row.get('objet_categorie'),
                "statut": row.get('statut_final'),
                "texte_original": row.get('texte_plainte'),
                "solution_suggeree": row.get('texte_solution')
            })
            
        return results

# Singleton: Pour éviter de recharger le lourd modèle IA à chaque requête HTTP
# Il est chargé une fois au démarrage de l'API.
vector_db = VectorSearchService()
