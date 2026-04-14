from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
from app.services.vector_service import vector_db
from app.services.llm_service import generate_solution_from_history

router = APIRouter(prefix="/search", tags=["RAG Search"])

class SearchRequest(BaseModel):
    texte_actuel: str
    categorie: Optional[str] = None
    gravite_max: Optional[str] = None

@router.post("/")
def search_similar(request: SearchRequest):
    # 1. Recherche Sémantique Faiss
    results = vector_db.search_similar(
        query=request.texte_actuel,
        top_k=3,
        category_filter=request.categorie
    )
    
    # 2. Génération de solution via LLM (Ollama)
    # Extraire uniquement les textes des solutions de l'historique pour le prompt
    historic_solutions_text = [res["solution_suggeree"] for res in results]
    
    # Si des résultats existent, on demande à Llama de rédiger une réponse
    generated_solution = "Aucune similarité trouvée."
    if historic_solutions_text:
        generated_solution = generate_solution_from_history(request.texte_actuel, historic_solutions_text)

    # 3. Réponse finale combinée (Génératif + Sources historiques)
    return {
        "message": generated_solution, # Solution générée pour la Section A du Front-end
        "resultats_trouves": len(results),
        "similar_claims": results      # Sources historiques pour la Section B du Front-end
    }
